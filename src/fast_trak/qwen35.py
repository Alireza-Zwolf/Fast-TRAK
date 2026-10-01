"""Qwen 3.5 specifics: the gated-delta-rule kernel and batching defaults.

Everything architecture-specific lives here so the rest of the package stays
model-agnostic.
"""

from __future__ import annotations

import importlib
import sys

MODEL_TYPES = frozenset({"qwen3_5", "qwen3_5_text"})
_MODELING_MODULE = "transformers.models.qwen3_5.modeling_qwen3_5"

# Qwen 3.5's DeltaNet layers process the sequence in chunks of this many
# tokens, so padding batches to a multiple of it adds no extra chunk work.
PAD_TO_MULTIPLE_OF = 64

# Padded-token budgets per batch at which throughput plateaus on a 46 GB
# NVIDIA L40S, keyed by the model-size tag in the checkpoint name.
TOKEN_BUDGETS = {"0.8B": 4096, "2B": 3072, "4B": 1536}


def token_budget(base_model: str) -> int | None:
    """Measured per-batch token budget for a Qwen 3.5 size, if known."""
    for size, budget in TOKEN_BUDGETS.items():
        if f"-{size}" in base_model:
            return budget
    return None


def gated_delta_rule_backend() -> str:
    """Which gated delta rule Qwen 3.5 will run: ``"fla"`` or ``"torch"``.

    Transformers wraps the selected implementation in closures, so the
    function's own module is not enough; the closure cells are searched for a
    callable that lives in the ``fla`` package.
    """
    modeling = importlib.import_module(_MODELING_MODULE)
    pending, seen = [modeling.torch_chunk_gated_delta_rule], set()
    while pending:
        function = pending.pop()
        if id(function) in seen:
            continue
        seen.add(id(function))
        if str(getattr(function, "__module__", "")).startswith("fla."):
            return "fla"
        for cell in getattr(function, "__closure__", None) or ():
            try:
                value = cell.cell_contents
            except ValueError:  # empty cell
                continue
            if callable(value):
                pending.append(value)
        if hasattr(function, "__wrapped__"):
            pending.append(function.__wrapped__)
    return "torch"


def enable_fla_kernel() -> None:
    """Route Qwen 3.5's gated delta rule to the FLA Triton kernel.

    Transformers picks the implementation once, while importing the Qwen 3.5
    modelling module, and only finds the kernel if ``fla.ops`` is already
    bound; otherwise it silently uses a pure-PyTorch fallback that loops over
    chunks in Python. Importing the kernel module first fixes the lookup.

    The kernel cannot be traced by ``torch.func``, so it is unusable under a
    vmap-based gradient computer. The hook-based computer never traces the
    model and can use it freely. Must be called before the model is built.

    Raises:
        ImportError: if ``fla`` is not installed.
        RuntimeError: if the modelling module was already imported with the
            fallback, at which point the choice is frozen for the process.
    """
    if _MODELING_MODULE in sys.modules and gated_delta_rule_backend() != "fla":
        raise RuntimeError(
            "Qwen 3.5 modelling code was imported before the FLA kernel was enabled; "
            "the gated-delta-rule choice is frozen to the PyTorch fallback for this process."
        )
    importlib.import_module("fla.ops.gated_delta_rule")
