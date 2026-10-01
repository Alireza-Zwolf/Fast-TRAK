"""Train one LoRA reference adapter: a minimal trainer for the quickstart.

TRAK attributes with respect to trained checkpoints, so it needs at least one
LoRA adapter fitted to the candidate pool (several seeds make a better
ensemble). Any trainer that saves a PEFT adapter works; this one is kept
deliberately small. Each example is rendered as

    Question: <question>\\nAnswer: <label><eos>

and only the answer tokens carry loss, matching the prompt FAST-TRAK scores.
"""

from __future__ import annotations

import argparse
import random

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from fast_trak import qwen35
from fast_trak.data import DEFAULT_PROMPT_TEMPLATE, read_examples
from fast_trak.models import load_tokenizer, model_type


def encode(example, tokenizer, max_length):
    prompt = tokenizer(
        DEFAULT_PROMPT_TEMPLATE.format(question=example.question), add_special_tokens=False
    )["input_ids"]
    answer = tokenizer(example.answer + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    prompt = prompt[-(max_length - len(answer)) :]
    return prompt + answer, [-100] * len(prompt) + answer


def collate(batch, pad_token_id, device):
    width = max(len(ids) for ids, _ in batch)
    input_ids = [ids + [pad_token_id] * (width - len(ids)) for ids, _ in batch]
    labels = [labels + [-100] * (width - len(labels)) for _, labels in batch]
    mask = [[1] * len(ids) + [0] * (width - len(ids)) for ids, _ in batch]
    return (
        torch.tensor(input_ids, device=device),
        torch.tensor(mask, device=device),
        torch.tensor(labels, device=device),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True, help="where the adapter is saved")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument(
        "--target_modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if model_type(args.base_model) in qwen35.MODEL_TYPES:
        try:
            qwen35.enable_fla_kernel()
        except (ImportError, RuntimeError):
            pass  # the PyTorch fallback is slower but equally correct

    tokenizer = load_tokenizer(args.base_model)
    examples = read_examples(args.train_file, args.labels)
    data = [encode(example, tokenizer, args.max_length) for example in examples]

    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.float32)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=args.target_modules,
            task_type="CAUSAL_LM",
        ),
    ).to(args.device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    use_amp = args.device.startswith("cuda")

    model.train()
    for epoch in range(args.epochs):
        random.shuffle(data)
        total = 0.0
        for start in range(0, len(data), args.batch_size):
            input_ids, mask, labels = collate(
                data[start : start + args.batch_size], tokenizer.pad_token_id, args.device
            )
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = model(input_ids=input_ids, attention_mask=mask, labels=labels).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total += loss.item() * len(input_ids)
        print(f"seed {args.seed} epoch {epoch + 1}/{args.epochs}: loss {total / len(data):.4f}")

    model.save_pretrained(args.out_dir)
    print(f"Saved adapter to {args.out_dir}")


if __name__ == "__main__":
    main()
