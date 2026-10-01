"""Mapping class labels to the token the model predicts first."""

from __future__ import annotations

from collections.abc import Sequence

from .data import DEFAULT_PROMPT_TEMPLATE


def _first_token_in_context(tokenizer, prefix: str, label: str) -> int:
    """First token contributed by ``label`` when it follows ``prefix``.

    Sub-word tokenisers can merge or split differently at a boundary, so the
    label is tokenised in its prompt context rather than in isolation.
    """
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    combined_ids = tokenizer(prefix + label, add_special_tokens=False)["input_ids"]
    shared = 0
    while (
        shared < min(len(prefix_ids), len(combined_ids))
        and prefix_ids[shared] == combined_ids[shared]
    ):
        shared += 1
    if shared >= len(combined_ids):
        raise ValueError(f"Label {label!r} contributes no token after the prompt")
    return combined_ids[shared]


def label_first_token_ids(
    tokenizer,
    labels: Sequence[str],
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
) -> list[int]:
    """Return each label's first answer token, requiring them to be distinct.

    The closed-label margin reads one logit per class at the first answer
    position, so two labels that begin with the same token are not separable
    there and are rejected.
    """
    if not labels:
        raise ValueError("At least one label is required")
    prefix = prompt_template.format(question="X")
    ids = [_first_token_in_context(tokenizer, prefix, label) for label in labels]
    if len(set(ids)) != len(ids):
        raise ValueError(
            "Labels must start with pairwise-distinct tokens, got "
            f"{dict(zip(labels, ids, strict=False))}. Rename the clashing labels."
        )
    return ids
