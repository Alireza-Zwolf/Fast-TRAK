from __future__ import annotations

import pytest
import torch

from fast_trak.outputs import AnswerMarginOutput, final_position_logits


class FixedLogits(torch.nn.Module):
    """Returns preset final-position logits and records how it was called."""

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.logits = logits
        self.kwargs = None

    def forward(self, input_ids, attention_mask, use_cache=True, logits_to_keep=0):
        self.kwargs = {"use_cache": use_cache, "logits_to_keep": logits_to_keep}
        return self.logits.unsqueeze(1)


class FullSequenceOnly(torch.nn.Module):
    def forward(self, input_ids, attention_mask):
        return torch.arange(12.0).reshape(1, 3, 4)


BATCH = (
    torch.zeros(2, 3, dtype=torch.long),
    torch.ones(2, 3, dtype=torch.long),
    torch.tensor([0, 2]),
)


def test_margin_and_loss_factor_match_the_closed_form():
    logits = torch.tensor([[0.0, 2.0, -1.0, 0.5, 9.0], [0.0, 0.3, 1.0, -2.0, 9.0]])
    model = FixedLogits(logits)
    margins, loss_factor = AnswerMarginOutput([1, 2, 3]).forward_batched(model, BATCH)

    label_logits = logits[:, 1:4]  # the vocabulary entry at index 4 is not a label
    for row, label in enumerate([0, 2]):
        others = torch.cat([label_logits[row, :label], label_logits[row, label + 1 :]])
        assert margins[row].item() == pytest.approx(
            (label_logits[row, label] - others.logsumexp(0)).item(), rel=1e-6
        )
        probability = torch.softmax(label_logits[row], 0)[label]
        assert loss_factor[row, 0].item() == pytest.approx(1 - probability.item(), rel=1e-6)
    assert loss_factor.shape == (2, 1) and not loss_factor.requires_grad


def test_loss_factor_survives_bfloat16_saturation():
    logits = torch.zeros(1, 4, dtype=torch.bfloat16)
    logits[0, 1] = 9.0  # p(correct) ~ 0.9998: 1 - p underflows to 0 in bfloat16
    batch = (BATCH[0][:1], BATCH[1][:1], torch.tensor([0]))
    _, loss_factor = AnswerMarginOutput([1, 2, 3]).forward_batched(FixedLogits(logits), batch)
    assert 1e-4 < loss_factor.item() < 1e-3


def test_requests_only_the_final_position_when_supported():
    model = FixedLogits(torch.zeros(2, 5))
    final_position_logits(model, BATCH[0], BATCH[1])
    assert model.kwargs == {"use_cache": False, "logits_to_keep": 1}


def test_falls_back_to_slicing_the_full_sequence():
    logits = final_position_logits(FullSequenceOnly(), BATCH[0][:1], BATCH[1][:1])
    assert logits.tolist() == [[8.0, 9.0, 10.0, 11.0]]


@pytest.mark.parametrize("ids", [[], [3, 3]])
def test_rejects_empty_or_duplicate_label_tokens(ids):
    with pytest.raises(ValueError, match="distinct"):
        AnswerMarginOutput(ids)
