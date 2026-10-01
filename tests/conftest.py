"""Shared fixtures: a tiny offline causal LM with LoRA adapters and a toy task."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

LABELS = ("cat", "dog", "bird")
WORDS = ["fur", "purr", "bark", "fetch", "wing", "nest", "tail", "paw", "seed", "bone"]


@dataclass
class TinyTask:
    base_model: str
    checkpoints: tuple[str, ...]
    candidates_file: str
    targets_file: str
    labels: tuple[str, ...] = LABELS


def _write_tsv(path: Path, rows: list[tuple[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["question", "answer"])
        writer.writerows(rows)


def _random_rows(rng: random.Random, count: int) -> list[tuple[str, str]]:
    return [
        (" ".join(rng.choices(WORDS, k=rng.randint(2, 9))), rng.choice(LABELS))
        for _ in range(count)
    ]


@pytest.fixture(scope="session")
def tiny_task(tmp_path_factory) -> TinyTask:
    """A 2-layer Llama, two LoRA checkpoints, 24 candidates and 5 targets."""
    root = tmp_path_factory.mktemp("tiny_task")
    specials = ["[PAD]", "[UNK]", "[EOS]"]
    vocabulary = specials + ["Question:", "Answer:"] + WORDS + list(LABELS)
    backend = Tokenizer(
        WordLevel({token: i for i, token in enumerate(vocabulary)}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]"
    )

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=len(vocabulary),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=64,
        pad_token_id=0,
        eos_token_id=2,
    )
    base_dir = root / "base"
    LlamaForCausalLM(config).save_pretrained(base_dir)
    tokenizer.save_pretrained(base_dir)

    checkpoints = []
    for seed in (1, 2):
        torch.manual_seed(seed)
        adapter = get_peft_model(
            LlamaForCausalLM.from_pretrained(base_dir),
            # init_lora_weights=False gives a non-zero B, hence non-trivial gradients.
            LoraConfig(
                r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], init_lora_weights=False
            ),
        )
        directory = root / f"adapter_{seed}"
        adapter.save_pretrained(directory)
        checkpoints.append(str(directory))

    rng = random.Random(0)
    candidates_file, targets_file = root / "candidates.tsv", root / "targets.tsv"
    _write_tsv(candidates_file, _random_rows(rng, 24))
    _write_tsv(targets_file, _random_rows(rng, 5))
    return TinyTask(str(base_dir), tuple(checkpoints), str(candidates_file), str(targets_file))
