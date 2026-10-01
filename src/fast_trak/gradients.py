"""Exact per-example LoRA gradients from one ordinary batched backward pass."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from trak.gradient_computers import AbstractGradientComputer

if TYPE_CHECKING:
    from .outputs import BatchedModelOutput


class BatchedLoRAGradientComputer(AbstractGradientComputer):
    """Per-example gradients of a scalar model output w.r.t. LoRA weights.

    For a linear layer ``y = W x`` the gradient of a per-example scalar ``f_b``
    with respect to ``W`` is the token-summed outer product

        d f_b / d W = sum_t  (d f_b / d y_{b,t})  x_{b,t}^T

    Forward hooks record each adapter layer's input ``x``; a tensor hook on its
    output records ``d f / d y`` while the summed output ``sum_b f_b`` is
    back-propagated once. Because examples in a batch do not interact, the
    rows of ``d (sum_b f_b) / d y`` are exactly the per-example output
    gradients, so the product above is the exact per-example gradient.

    This replaces ``torch.func.vmap`` over a functionalised model, which
    cannot trace architectures that use custom autograd kernels or
    data-dependent control flow. Only trainable LoRA ``A``/``B``
    :class:`torch.nn.Linear` weights are supported; anything else trainable is
    rejected up front rather than silently skipped.
    """

    def __init__(
        self,
        model: nn.Module,
        task: BatchedModelOutput,
        grad_dim: int,
        dtype: torch.dtype,
        device,
        grad_wrt: Iterable[str] | None = None,
    ) -> None:
        super().__init__(model=model, task=task, grad_dim=grad_dim, dtype=dtype, device=device)
        self.grad_wrt = set(grad_wrt) if grad_wrt is not None else None
        self.parameter_names: list[str] = []
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._hooked_modules: dict[nn.Module, str] = {}
        self._sample_grads: dict[str, Tensor] = {}
        self._loss_grad: Tensor | None = None
        self._capturing = False
        self._batch_size = 0
        self.load_model_params(model)

    # ------------------------------------------------------------------ setup

    def load_model_params(self, model: nn.Module) -> None:
        """Resolve the adapter layers to differentiate and (re)attach hooks."""
        self.model = model
        selected = OrderedDict(
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and (self.grad_wrt is None or name in self.grad_wrt)
        )
        if not selected:
            raise ValueError("No trainable parameters were selected for attribution")

        modules = dict(model.named_modules())
        hooked: dict[nn.Module, str] = {}
        unsupported: list[str] = []
        for name in selected:
            module_name, _, leaf = name.rpartition(".")
            module = modules.get(module_name)
            is_lora_linear = (
                leaf == "weight"
                and isinstance(module, nn.Linear)
                and ("lora_A." in name or "lora_B." in name)
            )
            if not is_lora_linear or module in hooked:
                unsupported.append(name)
                continue
            hooked[module] = name
        if unsupported:
            raise TypeError(
                "BatchedLoRAGradientComputer supports only LoRA A/B Linear weights; "
                f"unsupported trainable parameters: {unsupported}"
            )

        selected_dim = sum(parameter.numel() for parameter in selected.values())
        if selected_dim != self.grad_dim:
            raise ValueError(
                f"grad_dim={self.grad_dim:,} but the selected LoRA weights hold "
                f"{selected_dim:,} scalars"
            )

        self.parameter_names = list(selected)
        if hooked != self._hooked_modules:
            self.close()
            self._hooked_modules = hooked
            for module, name in hooked.items():
                self._handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def capture(module: nn.Module, args: tuple, output: Tensor) -> None:
            if not self._capturing:
                return
            if not args or not isinstance(args[0], Tensor) or not isinstance(output, Tensor):
                raise TypeError(f"Unexpected call signature for LoRA layer {name}")
            activation = args[0].detach()

            def accumulate(grad_output: Tensor) -> None:
                batch = self._batch_size
                if activation.shape[0] != batch or grad_output.shape[0] != batch:
                    raise RuntimeError(
                        f"LoRA layer {name} lost the batch dimension: "
                        f"input={tuple(activation.shape)}, grad_output={tuple(grad_output.shape)}, "
                        f"expected a leading dimension of {batch}"
                    )
                x = activation.reshape(batch, -1, activation.shape[-1])
                dy = grad_output.detach().reshape(batch, -1, grad_output.shape[-1])
                sample_grad = torch.einsum("bto,bti->boi", dy, x)
                # A layer invoked several times per forward contributes a sum.
                previous = self._sample_grads.get(name)
                self._sample_grads[name] = (
                    sample_grad if previous is None else previous + sample_grad
                )

            output.register_hook(accumulate)

        return capture

    # ------------------------------------------------------------- gradients

    def compute_per_sample_grad(self, batch: Iterable[Tensor]) -> OrderedDict[str, Tensor]:
        """Return ``{parameter name: [batch, out_features, in_features]}``."""
        batch = tuple(batch)
        self.reset()
        self._batch_size = int(batch[0].shape[0])
        self.model.zero_grad(set_to_none=True)
        self._capturing = True
        try:
            outputs, self._loss_grad = self.modelout_fn.forward_batched(self.model, batch)
            if outputs.shape != (self._batch_size,):
                raise ValueError(
                    f"Expected one model output per example, got shape {tuple(outputs.shape)}"
                )
            outputs.sum().backward()
        finally:
            self._capturing = False
            self.model.zero_grad(set_to_none=True)

        missing = [name for name in self.parameter_names if name not in self._sample_grads]
        if missing:
            raise RuntimeError(f"No per-example gradient was captured for: {missing}")
        return OrderedDict((name, self._sample_grads[name]) for name in self.parameter_names)

    def compute_loss_grad(self, batch: Iterable[Tensor]) -> Tensor:
        """Return the output-to-loss factor cached by the last forward pass."""
        if self._loss_grad is None:
            raise RuntimeError("compute_per_sample_grad must run before compute_loss_grad")
        return self._loss_grad

    # --------------------------------------------------------------- cleanup

    def reset(self) -> None:
        """Drop tensors cached from the previous batch (frees GPU memory)."""
        self._sample_grads = {}
        self._loss_grad = None

    def close(self) -> None:
        """Remove every hook this computer installed on the model."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._hooked_modules = {}
