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


def configure_fla_kernel(base_model: str, fla_kernel: bool | None = None) -> None:
    """Select Qwen 3.5's gated-delta-rule kernel; a no-op for other models.

    ``None`` uses the FLA Triton kernel when it is installed, ``True`` requires
    it and ``False`` keeps the PyTorch fallback. The choice is fixed the first
    time Qwen 3.5 is loaded in a process, so call this before building any
    model, including one built for training.
    """
    if model_type(base_model) not in qwen35.MODEL_TYPES:
        if fla_kernel:
            raise ValueError("The FLA kernel only applies to Qwen 3.5 models")
        return
    if fla_kernel is False:
        return
    try:
        qwen35.enable_fla_kernel()
    except (ImportError, RuntimeError) as error:
        if fla_kernel:
            raise
        logger.warning(
            "Qwen 3.5 is running the slow pure-PyTorch gated delta rule (%s). "
            "Install fla-core to use the FLA Triton kernel.",
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
    configure_fla_kernel(base_model, fla_kernel)

    base = AutoModelForCausalLM.from_pretrained(base_model, dtype=DTYPES[dtype])
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=True)
    model.to(device)
    model.eval()
    if model_type(base_model) in qwen35.MODEL_TYPES:
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
