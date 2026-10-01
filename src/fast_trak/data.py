"""Reading, tokenising and batching prompts for attribution."""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_TEMPLATE = "Question: {question}\nAnswer: "


@dataclass(frozen=True)
class Example:
    """One labelled text. ``row`` is its 0-based data row in the source file."""

    row: int
    question: str
    answer: str


@dataclass(frozen=True)
class TokenizedExample:
    """A tokenised prompt. ``index`` is the example's row in the TRAK store."""

    index: int
    input_ids: list[int]
    label: int


def read_examples(
    path: str,
    labels: Sequence[str],
    question_col: str = "question",
    answer_col: str = "answer",
    drop_unknown_labels: bool = False,
    limit: int | None = None,
) -> list[Example]:
    """Load ``(question, answer)`` rows from a headed TSV or CSV file.

    Rows whose answer is outside ``labels`` raise by default. With
    ``drop_unknown_labels`` they are skipped instead, and every kept example
    still carries its original ``row`` so scores stay traceable to the file.
    ``limit`` keeps the first N usable rows.
    """
    delimiter = "," if Path(path).suffix.lower() == ".csv" else "\t"
    label_set = set(labels)
    examples: list[Example] = []
    unknown: list[int] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        missing = {question_col, answer_col} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} lacks column(s) {sorted(missing)}; found {reader.fieldnames}")
        for row, record in enumerate(reader):
            question = (record[question_col] or "").strip()
            answer = (record[answer_col] or "").strip()
            if answer not in label_set:
                unknown.append(row)
                continue
            examples.append(Example(row=row, question=question, answer=answer))

    if unknown and not drop_unknown_labels:
        raise ValueError(
            f"{path}: {len(unknown)} row(s) have an answer outside the label set "
            f"(first rows: {unknown[:5]}). Fix the file or pass drop_unknown_labels."
        )
    if unknown:
        logger.warning("%s: dropped %d row(s) with an unknown label", path, len(unknown))
    if limit is not None:
        examples = examples[:limit]
    if not examples:
        raise ValueError(f"{path} contains no usable rows")
    return examples


class PromptDataset(Dataset):
    """Prompts tokenised once, without any global padding.

    Each prompt ends exactly where its answer would begin, so the model's
    final-position logits are the first-answer-token prediction. Prompts longer
    than ``max_length`` are truncated from the left to preserve that position.
    """

    def __init__(
        self,
        examples: Sequence[Example],
        tokenizer,
        labels: Sequence[str],
        max_length: int,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    ) -> None:
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")
        label_to_index = {label: index for index, label in enumerate(labels)}
        self.samples: list[TokenizedExample] = []
        for index, example in enumerate(examples):
            if example.answer not in label_to_index:
                raise ValueError(
                    f"Example {index} has answer {example.answer!r}, absent from the label set"
                )
            ids = tokenizer(
                prompt_template.format(question=example.question),
                add_special_tokens=False,
            )["input_ids"]
            self.samples.append(
                TokenizedExample(
                    index=index,
                    input_ids=list(ids[-max_length:]),
                    label=label_to_index[example.answer],
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> TokenizedExample:
        return self.samples[index]

    def length(self, index: int) -> int:
        return len(self.samples[index].input_ids)


def padded_length(length: int, pad_to_multiple_of: int | None) -> int:
    """Round ``length`` up to the padding multiple (no-op when unset)."""
    if not pad_to_multiple_of:
        return length
    return -(-length // pad_to_multiple_of) * pad_to_multiple_of


class TokenBudgetBatchSampler(Sampler[list[int]]):
    """Deterministic length-sorted batches bounded by padded tokens.

    A batch costs ``len(batch) * padded_length(longest prompt)``, exactly the
    size of the tensor :class:`LeftPadCollator` builds. Batches grow until the
    next (equal or longer) prompt would exceed ``max_tokens`` or
    ``max_batch_size``; a prompt that alone exceeds the budget forms a batch of
    one. Short prompts therefore run in large batches and long ones in small
    batches, with almost no padding in either.
    """

    def __init__(
        self,
        dataset: PromptDataset,
        max_tokens: int,
        max_batch_size: int | None = None,
        pad_to_multiple_of: int | None = None,
    ) -> None:
        max_batch_size = max_batch_size or max(len(dataset), 1)
        if max_tokens <= 0 or max_batch_size <= 0:
            raise ValueError(
                "max_tokens and max_batch_size must be positive, "
                f"got {max_tokens} and {max_batch_size}"
            )
        order = sorted(range(len(dataset)), key=lambda i: (dataset.length(i), i))
        self.batches: list[list[int]] = []
        current: list[int] = []
        for index in order:
            width = padded_length(dataset.length(index), pad_to_multiple_of)
            grown = len(current) + 1
            if current and (grown > max_batch_size or grown * width > max_tokens):
                self.batches.append(current)
                current = []
            current.append(index)
        if current:
            self.batches.append(current)

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


class LeftPadCollator:
    """Left-pad a batch so every prompt ends at the final position.

    Returns ``(store indices, input_ids, attention_mask, labels)``.
    """

    def __init__(self, pad_token_id: int, pad_to_multiple_of: int | None = None) -> None:
        if pad_token_id is None:
            raise ValueError("The tokenizer must define a pad token")
        if pad_to_multiple_of is not None and pad_to_multiple_of <= 0:
            raise ValueError("pad_to_multiple_of must be positive")
        self.pad_token_id = int(pad_token_id)
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, samples: Sequence[TokenizedExample]):
        width = padded_length(
            max(len(sample.input_ids) for sample in samples), self.pad_to_multiple_of
        )
        input_ids, attention_mask = [], []
        for sample in samples:
            pad = width - len(sample.input_ids)
            input_ids.append([self.pad_token_id] * pad + sample.input_ids)
            attention_mask.append([0] * pad + [1] * len(sample.input_ids))
        return (
            torch.tensor([sample.index for sample in samples], dtype=torch.long),
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor([sample.label for sample in samples], dtype=torch.long),
        )


def build_loader(
    examples: Sequence[Example],
    tokenizer,
    labels: Sequence[str],
    *,
    max_length: int,
    max_tokens_per_batch: int,
    max_batch_size: int | None = None,
    pad_to_multiple_of: int | None = None,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
) -> DataLoader:
    """Tokenise ``examples`` and wrap them in a token-budget batched loader."""
    dataset = PromptDataset(examples, tokenizer, labels, max_length, prompt_template)
    sampler = TokenBudgetBatchSampler(
        dataset, max_tokens_per_batch, max_batch_size, pad_to_multiple_of
    )
    collator = LeftPadCollator(tokenizer.pad_token_id, pad_to_multiple_of)
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=collator)
