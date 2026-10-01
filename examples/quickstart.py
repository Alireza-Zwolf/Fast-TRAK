"""FAST-TRAK quickstart on the AG News topic-classification benchmark.

Answers one question: *which training examples made the model classify this
test headline the way it did?*

    python examples/quickstart.py

The script

1. downloads AG News and keeps 4,000 training articles (the candidates) and
   100 test articles (the targets);
2. fine-tunes two small LoRA adapters on the candidates;
3. scores every candidate against every target with FAST-TRAK;
4. prints the most helpful and most harmful training articles for one test
   article, and how often the top-ranked articles share the target's topic.

It needs one CUDA GPU and takes a few minutes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from fast_trak import TrakConfig, featurize, open_session, score
from fast_trak.data import DEFAULT_PROMPT_TEMPLATE, read_examples
from fast_trak.models import configure_fla_kernel, load_tokenizer

LABELS = ("World", "Sports", "Business", "Sci/Tech")
MAX_LENGTH = 128


def write_split(split: str, count: int, path: str, seed: int) -> None:
    """Sample ``count`` AG News articles into a question/answer TSV."""
    source = hf_hub_download("SetFit/ag_news", f"{split}.jsonl", repo_type="dataset")
    with open(source, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["question", "answer"])
        for row in random.Random(seed).sample(rows, count):
            writer.writerow([" ".join(row["text"].split()), row["label_text"]])


def train_adapter(base_model: str, train_file: str, out_dir: str, seed: int) -> None:
    """One epoch of LoRA fine-tuning on ``Question: <text>\\nAnswer: <label>``."""
    torch.manual_seed(seed)
    tokenizer = load_tokenizer(base_model)
    data = []
    for example in read_examples(train_file, LABELS):
        prompt = tokenizer(DEFAULT_PROMPT_TEMPLATE.format(question=example.question))["input_ids"]
        answer = tokenizer(example.answer + tokenizer.eos_token)["input_ids"]
        prompt = prompt[-(MAX_LENGTH - len(answer)) :]
        data.append((prompt + answer, [-100] * len(prompt) + answer))  # loss on the answer only
    random.Random(seed).shuffle(data)

    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32)
    lora = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lora).cuda().train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)

    for start in range(0, len(data), 16):
        batch = data[start : start + 16]
        width = max(len(ids) for ids, _ in batch)
        pad = tokenizer.pad_token_id
        input_ids = torch.tensor([ids + [pad] * (width - len(ids)) for ids, _ in batch]).cuda()
        labels = torch.tensor([lab + [-100] * (width - len(lab)) for _, lab in batch]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=input_ids, attention_mask=input_ids != pad, labels=labels).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    model.save_pretrained(out_dir)
    print(f"Trained adapter {seed}: final batch loss {loss.item():.3f}")


def report(scores: np.ndarray, candidates, targets) -> None:
    """Print what the scores say, for one target and on average."""
    labels = np.array([c.answer for c in candidates])
    top10 = np.argsort(-scores, axis=0)[:10]  # [10, targets]
    agreement = np.mean([(labels[top10[:, j]] == t.answer).mean() for j, t in enumerate(targets)])
    print(f"\nTop-10 training articles that share the test article's topic: {agreement:.0%}")
    print(f"(picking training articles at random would give {1 / len(LABELS):.0%})")

    target = targets[0]
    order = np.argsort(-scores[:, 0])
    print(f"\nTest article [{target.answer}]: {target.question[:110]}")
    for title, rows in (("Most helpful", order[:3]), ("Most harmful", order[::-1][:3])):
        print(f"\n{title} training articles:")
        for row in rows:
            c = candidates[row]
            print(f"  {scores[row, 0]:+.4f}  [{c.answer}] {c.question[:90]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base_model", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--out", default="quickstart_out")
    args = parser.parse_args()

    configure_fla_kernel(args.base_model)  # before any model is built
    os.makedirs(args.out, exist_ok=True)
    candidates_file = os.path.join(args.out, "candidates.tsv")
    targets_file = os.path.join(args.out, "targets.tsv")
    write_split("train", 4000, candidates_file, seed=0)
    write_split("test", 100, targets_file, seed=0)

    checkpoints = []
    for seed in (0, 1):
        checkpoints.append(os.path.join(args.out, f"adapter_{seed}"))
        if not os.path.isdir(checkpoints[-1]):
            train_adapter(args.base_model, candidates_file, checkpoints[-1], seed)

    config = TrakConfig(
        base_model=args.base_model,
        checkpoints=tuple(checkpoints),
        candidates_file=candidates_file,
        labels=LABELS,
        save_dir=os.path.join(args.out, "trak_store"),
        max_length=MAX_LENGTH,
        # The projection must be much smaller than the number of candidates.
        projector="basic",
        proj_dim=64,
        num_projections=4,
    )
    session = open_session(config)
    featurize(config, session=session)
    scores = score(config, targets_file, session=session)  # [candidates, targets]

    report(scores, session.candidates, session.read(targets_file))


if __name__ == "__main__":
    main()
