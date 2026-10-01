from __future__ import annotations

import random

import pytest
import torch

from fast_trak.data import (
    Example,
    LeftPadCollator,
    PromptDataset,
    TokenBudgetBatchSampler,
    padded_length,
    read_examples,
)
from fast_trak.labels import label_first_token_ids


class WordTokenizer:
    """One token per whitespace-separated word; IDs are stable per word."""

    pad_token_id = 0

    def __init__(self):
        self.vocabulary = {}

    def __call__(self, text, add_special_tokens=False):
        ids = [self.vocabulary.setdefault(word, len(self.vocabulary) + 1) for word in text.split()]
        return {"input_ids": ids}


def examples(*pairs):
    return [Example(row, question, answer) for row, (question, answer) in enumerate(pairs)]


def dataset(pairs, labels, max_length=1000):
    return PromptDataset(examples(*pairs), WordTokenizer(), labels, max_length, "{question}")


def test_read_examples_keeps_original_rows_when_dropping_unknown_labels(tmp_path):
    path = tmp_path / "data.tsv"
    path.write_text("answer\tquestion\nA\tfirst\nZ\tsecond\nB\t third \n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside the label set"):
        read_examples(str(path), ["A", "B"])
    kept = read_examples(str(path), ["A", "B"], drop_unknown_labels=True)
    assert kept == [Example(0, "first", "A"), Example(2, "third", "B")]
    assert read_examples(str(path), ["A", "B"], drop_unknown_labels=True, limit=1) == kept[:1]


def test_read_examples_reports_missing_columns(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("text,answer\nhello,A\n", encoding="utf-8")
    with pytest.raises(ValueError, match="question"):
        read_examples(str(path), ["A"])
    assert read_examples(str(path), ["A"], question_col="text") == [Example(0, "hello", "A")]


def test_long_prompts_are_truncated_from_the_left():
    data = dataset([("a b c d e", "A")], ["A"], max_length=2)
    tokenizer = WordTokenizer()
    assert data[0].input_ids == tokenizer("a b c d e")["input_ids"][-2:]


def test_collator_left_pads_and_keeps_store_indices():
    data = dataset([("one two three", "A"), ("one", "B"), ("one two", "A")], ["A", "B"])
    indices, input_ids, attention_mask, labels = LeftPadCollator(0)([data[1], data[2]])

    assert indices.tolist() == [1, 2]
    assert attention_mask.tolist() == [[0, 1], [1, 1]]
    assert torch.all(input_ids[attention_mask == 0] == 0)
    assert labels.tolist() == [1, 0]


def test_collator_rounds_the_batch_width():
    data = dataset([("one", "A"), ("one two three four", "A")], ["A"])
    assert LeftPadCollator(0, pad_to_multiple_of=8)(data.samples)[1].shape == (2, 8)


def test_token_budget_batches_cover_every_index_once_within_budget():
    rng = random.Random(0)
    data = dataset([(" ".join(["w"] * rng.randint(1, 200)), "A") for _ in range(500)], ["A"], 150)
    sampler = TokenBudgetBatchSampler(
        data, max_tokens=1024, max_batch_size=12, pad_to_multiple_of=64
    )
    batches = list(sampler)

    flat = [index for batch in batches for index in batch]
    assert sorted(flat) == list(range(len(data)))
    lengths = [data.length(index) for index in flat]
    assert lengths == sorted(lengths)
    for batch in batches:
        width = padded_length(max(data.length(index) for index in batch), 64)
        assert len(batch) <= 12 and len(batch) * width <= 1024
        assert LeftPadCollator(0, 64)([data[i] for i in batch])[1].numel() == len(batch) * width
    assert len(sampler) == len(batches)


def test_an_oversized_prompt_forms_its_own_batch():
    data = dataset([("w " * 100, "A"), ("w", "A"), ("w", "A")], ["A"])
    sampler = TokenBudgetBatchSampler(data, max_tokens=128, pad_to_multiple_of=64)
    assert list(sampler) == [[1, 2], [0]]


def test_label_tokens_are_read_in_prompt_context_and_must_be_distinct():
    tokenizer = WordTokenizer()
    ids = label_first_token_ids(tokenizer, ["yes", "no way"], "Q: {question} A: ")
    assert ids == [tokenizer("yes")["input_ids"][0], tokenizer("no")["input_ids"][0]]
    with pytest.raises(ValueError, match="pairwise-distinct"):
        label_first_token_ids(tokenizer, ["no way", "no chance"], "Q: {question} A: ")
