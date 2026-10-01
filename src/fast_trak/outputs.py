"""Model-output functions evaluated on an ordinary (non-functional) batch."""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Sequence
from typing import Protocol

import torch
from torch import Tensor
from trak.modelout_functions import AbstractModelOutput


class BatchedModelOutput(Protocol):
    """What :class:`~fast_trak.gradients.BatchedLoRAGradientComputer` calls.

    ``forward_batched`` returns one differentiable scalar per example and the
    matching (detached) output-to-loss factor of shape ``[batch, 1]``, the
    ``Q`` term of the TRAK estimator.
    """

    def forward_batched(
        self, model: torch.nn.Module, batch: Iterable[Tensor]
    ) -> tuple[Tensor, Tensor]: ...


def final_position_logits(
    model: torch.nn.Module, input_ids: Tensor, attention_mask: Tensor
) -> Tensor:
    """Vocabulary logits at the last position of a left-padded batch.

    When the model supports it, ``logits_to_keep=1`` restricts the language
    modelling head to that single position instead of the whole sequence.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    parameters = inspect.signature(base.forward).parameters
    kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
    for name, value in (("use_cache", False), ("logits_to_keep", 1)):
        if name in parameters:
            kwargs[name] = value
    outputs = model(**kwargs)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    return logits[:, -1, :]


class AnswerMarginOutput(AbstractModelOutput):
    """Closed-label margin at the first answer token of a causal LM.

    For a prompt that ends where the answer begins, with ``z_c`` the logit of
    the first token of label ``c`` and ``y`` the correct label:

        f = z_y - logsumexp_{c != y} z_c            (model output)
        Q = 1 - softmax(z)_y                        (output-to-loss factor)

    The softmax runs over the label tokens only, which makes this the
    multi-class margin of the TRAK paper for a classifier read off a
    generative model.
    """

    def __init__(self, label_token_ids: Sequence[int]) -> None:
        super().__init__()
        ids = list(label_token_ids)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"label token IDs must be non-empty and distinct, got {ids}")
        self.label_token_ids = torch.tensor(ids, dtype=torch.long)

    def forward_batched(
        self, model: torch.nn.Module, batch: Iterable[Tensor]
    ) -> tuple[Tensor, Tensor]:
        input_ids, attention_mask, labels = batch
        logits = final_position_logits(model, input_ids, attention_mask)
        label_logits = logits.index_select(-1, self.label_token_ids.to(logits.device))

        rows = torch.arange(label_logits.shape[0], device=label_logits.device)
        is_correct = torch.nn.functional.one_hot(labels, label_logits.shape[-1]).bool()
        correct = label_logits[rows, labels]
        others = label_logits.masked_fill(is_correct, -torch.inf).logsumexp(dim=-1)
        margins = correct - others

        # Q is taken in float32: in bfloat16, 1 - p rounds to exactly zero once
        # p exceeds ~0.998, which would zero the score of every confidently
        # fitted training example.
        probabilities = torch.softmax(label_logits.float(), dim=-1)[rows, labels]
        return margins, (1.0 - probabilities).detach().unsqueeze(-1)

    # TRAK's abstract interface is per-example and functional. Reaching it with
    # this object means the wrong gradient computer was configured, so fail
    # loudly instead of falling back to a slow path.
    def get_output(self, *args, **kwargs):  # pragma: no cover
        raise RuntimeError("AnswerMarginOutput requires BatchedLoRAGradientComputer")

    def get_out_to_loss_grad(self, *args, **kwargs):  # pragma: no cover
        raise RuntimeError("The output-to-loss factor is cached by BatchedLoRAGradientComputer")
