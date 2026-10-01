"""Loading a base causal LM with its LoRA adapters."""

from __future__ import annotations

import logging

import torch
from peft import PeftModel
from peft.utils import load_peft_weights, set_peft_model_state_dict
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from . import qwen35

logger = logging.getLogger(__name__)

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def model_type(base_model: str) -> str:
    """The Transformers ``model_type`` of a checkpoint (reads only its config)."""
    return AutoConfig.from_pretrained(base_model).model_type


def _configure_fla_kernel(is_qwen35: bool, fla_kernel: bool | None) -> None:
    """Apply the ``fla_kernel`` setting: ``True`` requires it, ``None`` tries it."""
    if fla_kernel is False or not is_qwen35:
        if fla_kernel and not is_qwen35:
            raise ValueError("The FLA kernel only applies to Qwen 3.5 models")
        return
    try:
        qwen35.enable_fla_kernel()
    except (ImportError, RuntimeError) as error:
        if fla_kernel:
            raise
        logger.warning(
            "Qwen 3.5 is running the slow pure-PyTorch gated delta rule (%s). "
            "Install fast-trak[qwen35] to use the FLA Triton kernel.",
            error,
        )


def load_tokenizer(base_model: str):
    """Load the tokenizer, falling back to EOS as the padding token."""
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    base_model: str,
    adapter_dir: str,
    device: str,
    dtype: str = "bfloat16",
    fla_kernel: bool | None = None,
) -> PeftModel:
    """Load ``base_model`` with the LoRA adapter in ``adapter_dir``, in eval mode.

    Only the adapter weights are left trainable, which is what selects the
    parameters attribution differentiates. ``fla_kernel`` controls Qwen 3.5's
    gated-delta-rule kernel: ``None`` enables it when available, ``True``
    requires it, ``False`` keeps the PyTorch fallback.
    """
    if dtype not in DTYPES:
        raise ValueError(f"Unsupported dtype {dtype!r}; choose from {sorted(DTYPES)}")
    is_qwen35 = model_type(base_model) in qwen35.MODEL_TYPES
    _configure_fla_kernel(is_qwen35, fla_kernel)

    base = AutoModelForCausalLM.from_pretrained(base_model, dtype=DTYPES[dtype])
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=True)
    model.to(device)
    model.eval()
    if is_qwen35:
        logger.info("Qwen 3.5 gated delta rule: %s", qwen35.gated_delta_rule_backend())
    return model


def load_adapter_weights(model: PeftModel, adapter_dir: str, device: str) -> None:
    """Swap another checkpoint's adapter weights into an already-built model."""
    state = load_peft_weights(adapter_dir, device=device)
    result = set_peft_model_state_dict(model, state)
    unexpected = getattr(result, "unexpected_keys", None)
    if unexpected:
        raise RuntimeError(f"Unexpected adapter keys in {adapter_dir}: {unexpected}")


def trainable_parameter_names(model: torch.nn.Module) -> list[str]:
    """Names of the parameters attribution differentiates, in model order."""
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]
