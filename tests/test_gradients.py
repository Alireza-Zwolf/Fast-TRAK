"""The hook-based gradients must equal explicit per-example autograd."""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
from torch import nn

from fast_trak.data import LeftPadCollator, PromptDataset, read_examples
from fast_trak.gradients import BatchedLoRAGradientComputer
from fast_trak.labels import label_first_token_ids
from fast_trak.models import load_model, load_tokenizer, trainable_parameter_names
from fast_trak.outputs import AnswerMarginOutput


class ToyLoRAModel(nn.Module):
    """A frozen linear layer with a LoRA update, optionally applied twice."""

    def __init__(self, reuse_adapter: bool = False):
        super().__init__()
        self.frozen = nn.Linear(5, 5, bias=False)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(5, 2, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(2, 5, bias=False)})
        self.frozen.weight.requires_grad_(False)
        self.reuse_adapter = reuse_adapter

    def layer(self, x):
        return self.frozen(x) + 1.7 * self.lora_B["default"](self.lora_A["default"](x))

    def forward(self, x):
        hidden = self.layer(x)
        return self.layer(torch.tanh(hidden)) if self.reuse_adapter else hidden


class ToyOutput:
    def forward_batched(self, model, batch):
        (x,) = batch
        outputs = model(x).square().sum(dim=(1, 2))
        return outputs, torch.sigmoid(-outputs).detach().unsqueeze(-1)


def loop_per_example_grads(outputs_for, model, batch_size):
    """Reference gradients: one autograd call per example."""
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    rows = OrderedDict((name, []) for name, _ in named)
    for index in range(batch_size):
        grads = torch.autograd.grad(outputs_for(index), [p for _, p in named])
        for (name, _), grad in zip(named, grads, strict=False):
            rows[name].append(grad)
    return OrderedDict((name, torch.stack(grads)) for name, grads in rows.items())


def make_computer(model, task):
    grad_dim = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return BatchedLoRAGradientComputer(model, task, grad_dim, torch.float64, "cpu")


@pytest.mark.parametrize("reuse_adapter", [False, True])
def test_hook_grads_equal_explicit_autograd(reuse_adapter):
    torch.manual_seed(7)
    model = ToyLoRAModel(reuse_adapter).double()
    x = torch.randn(3, 4, 5, dtype=torch.double)
    expected = loop_per_example_grads(lambda i: model(x[i : i + 1]).square().sum(), model, len(x))

    computer = make_computer(model, ToyOutput())
    actual = computer.compute_per_sample_grad((x,))

    assert list(actual) == list(expected)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-10, atol=1e-10)
    outputs = model(x).square().sum(dim=(1, 2))
    torch.testing.assert_close(computer.compute_loss_grad((x,))[:, 0], torch.sigmoid(-outputs))


def test_hook_grads_equal_autograd_on_a_padded_transformer(tiny_task):
    """End to end on a real LoRA-adapted causal LM with left-padded prompts."""
    tokenizer = load_tokenizer(tiny_task.base_model)
    model = load_model(tiny_task.base_model, tiny_task.checkpoints[0], "cpu", "float32").double()
    task = AnswerMarginOutput(label_first_token_ids(tokenizer, tiny_task.labels))
    examples = read_examples(tiny_task.candidates_file, tiny_task.labels, limit=6)
    dataset = PromptDataset(examples, tokenizer, tiny_task.labels, max_length=64)
    _, input_ids, attention_mask, labels = LeftPadCollator(tokenizer.pad_token_id)(dataset.samples)
    assert attention_mask.min() == 0, "the batch should contain padding"

    def margin(index):
        row = slice(index, index + 1)
        return task.forward_batched(model, (input_ids[row], attention_mask[row], labels[row]))[
            0
        ].sum()

    expected = loop_per_example_grads(margin, model, len(examples))
    computer = make_computer(model, task)
    actual = computer.compute_per_sample_grad((input_ids, attention_mask, labels))

    assert list(actual) == trainable_parameter_names(model)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-8, atol=1e-10)
    computer.close()


def test_rejects_trainable_parameters_that_are_not_lora():
    model = ToyLoRAModel()
    model.frozen.weight.requires_grad_(True)
    with pytest.raises(TypeError, match="frozen.weight"):
        make_computer(model, ToyOutput())


def test_rejects_a_grad_dim_that_does_not_match_the_adapter():
    with pytest.raises(ValueError, match="grad_dim"):
        BatchedLoRAGradientComputer(ToyLoRAModel(), ToyOutput(), 1, torch.float32, "cpu")


def test_close_removes_hooks_and_reset_drops_cached_tensors():
    model = ToyLoRAModel()
    computer = make_computer(model, ToyOutput())
    computer.compute_per_sample_grad((torch.randn(2, 3, 5),))
    computer.reset()
    with pytest.raises(RuntimeError, match="compute_per_sample_grad must run"):
        computer.compute_loss_grad(())
    computer.close()
    assert all(not module._forward_hooks for module in model.modules())
