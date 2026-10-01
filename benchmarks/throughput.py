"""Benchmark batched hook gradients against a per-example autograd loop.

Measures, on a real LoRA-adapted model and your own prompts:

* throughput (examples per second) of ``BatchedLoRAGradientComputer`` versus
  one ``autograd.grad`` call per example, which is the only other way to get
  exact per-example gradients when ``torch.func.vmap`` cannot trace the model;
* agreement between the two (relative L2 error and cosine similarity).

Example:

    python benchmarks/throughput.py \\
        --base_model Qwen/Qwen3.5-0.8B-Base --checkpoint path/to/adapter \\
        --candidates candidates.tsv --labels yes no --num_examples 512
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from fast_trak import qwen35
from fast_trak.data import DEFAULT_PROMPT_TEMPLATE, LeftPadCollator, build_loader, read_examples
from fast_trak.gradients import BatchedLoRAGradientComputer
from fast_trak.labels import label_first_token_ids
from fast_trak.models import load_model, load_tokenizer, model_type
from fast_trak.outputs import AnswerMarginOutput


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--checkpoint", required=True, help="LoRA adapter directory")
    parser.add_argument(
        "--candidates", required=True, help="headed TSV/CSV with question/answer columns"
    )
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--num_examples", type=int, default=512)
    parser.add_argument(
        "--loop_examples",
        type=int,
        default=64,
        help="examples timed with the per-example loop (it is slow)",
    )
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_tokens_per_batch", type=int, default=None)
    parser.add_argument("--model_dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--fla_kernel", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--drop_unknown_labels", action="store_true")
    parser.add_argument("--json", default=None, help="also write the results to this file")
    return parser


def flatten(grads) -> torch.Tensor:
    return torch.cat([g.reshape(g.shape[0], -1) for g in grads.values()], dim=1).float()


def synchronized_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def main() -> None:
    args = build_parser().parse_args()
    device = "cuda"
    is_qwen35 = model_type(args.base_model) in qwen35.MODEL_TYPES
    padding = qwen35.PAD_TO_MULTIPLE_OF if is_qwen35 else None
    budget = (
        args.max_tokens_per_batch
        or (qwen35.token_budget(args.base_model) if is_qwen35 else None)
        or 1024
    )

    tokenizer = load_tokenizer(args.base_model)
    model = load_model(args.base_model, args.checkpoint, device, args.model_dtype, args.fla_kernel)
    task = AnswerMarginOutput(label_first_token_ids(tokenizer, args.labels))
    examples = read_examples(
        args.candidates,
        args.labels,
        drop_unknown_labels=args.drop_unknown_labels,
        limit=args.num_examples,
    )
    loader = build_loader(
        examples,
        tokenizer,
        args.labels,
        max_length=args.max_length,
        max_tokens_per_batch=budget,
        pad_to_multiple_of=padding,
        prompt_template=DEFAULT_PROMPT_TEMPLATE,
    )
    batches = [tuple(t.to(device) for t in batch) for batch in loader]
    parameters = [p for p in model.parameters() if p.requires_grad]
    grad_dim = sum(p.numel() for p in parameters)
    computer = BatchedLoRAGradientComputer(model, task, grad_dim, torch.float32, device)

    # Warm-up: compiles kernels and allocates caches outside the timed region.
    computer.compute_per_sample_grad(batches[0][1:])
    computer.reset()

    torch.cuda.reset_peak_memory_stats()
    start = synchronized_time()
    for _, *batch in batches:
        computer.compute_per_sample_grad(batch)
    batched_seconds = synchronized_time() - start
    peak_gib = torch.cuda.max_memory_allocated() / 2**30

    # Per-example loop on a spread of lengths, each prompt with only its own padding.
    dataset, collate = loader.dataset, LeftPadCollator(tokenizer.pad_token_id, padding)
    step = max(len(dataset) // args.loop_examples, 1)
    loop_indices = list(range(0, len(dataset), step))[: args.loop_examples]
    position = {int(i): (b, r) for b, batch in enumerate(batches) for r, i in enumerate(batch[0])}

    errors, cosines, loop_seconds = [], [], 0.0
    for index in loop_indices:
        single = tuple(t.to(device) for t in collate([dataset[index]])[1:])
        start = synchronized_time()
        margin, _ = task.forward_batched(model, single)
        loop_grad = torch.autograd.grad(margin.sum(), parameters)
        loop_seconds += synchronized_time() - start

        batch_index, row = position[index]
        hooked = flatten(computer.compute_per_sample_grad(batches[batch_index][1:]))[row]
        reference = torch.cat([g.reshape(-1) for g in loop_grad]).float()
        errors.append(((hooked - reference).norm() / reference.norm()).item())
        cosines.append(torch.nn.functional.cosine_similarity(hooked, reference, dim=0).item())
    computer.close()

    results = {
        "base_model": args.base_model,
        "gpu": torch.cuda.get_device_name(0),
        "model_dtype": args.model_dtype,
        "gated_delta_rule": qwen35.gated_delta_rule_backend() if is_qwen35 else None,
        "lora_grad_dim": grad_dim,
        "examples": len(examples),
        "batches": len(batches),
        "token_budget": budget,
        "batched_examples_per_second": round(len(examples) / batched_seconds, 2),
        "loop_examples_per_second": round(len(loop_indices) / loop_seconds, 2),
        "speedup": round((len(examples) / batched_seconds) / (len(loop_indices) / loop_seconds), 2),
        "peak_gpu_gib": round(peak_gib, 2),
        "max_relative_l2_error": max(errors),
        "min_cosine_similarity": min(cosines),
    }
    print(json.dumps(results, indent=2))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
