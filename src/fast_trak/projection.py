"""Random projectors sized to the adapter gradient."""

from __future__ import annotations

import logging

import torch
from trak.projectors import AbstractProjector, BasicProjector, CudaProjector, ProjectionType

logger = logging.getLogger(__name__)

PROJECTOR_KINDS = ("auto", "cuda", "basic")
_CUDA_DIM_MULTIPLE = 512  # fast_jl projects in blocks of this many dimensions


def build_projector(
    grad_dim: int,
    proj_dim: int,
    seed: int,
    device: str,
    kind: str = "auto",
    max_batch_size: int = 32,
) -> AbstractProjector:
    """Build a Rademacher projector for a ``grad_dim``-dimensional gradient.

    TRAK's own auto-configuration sizes the projector from *all* model
    parameters, frozen base weights included, and so picks a chunked layout
    that does not match a LoRA-only gradient. Building the projector here,
    from the adapter dimension, avoids that and keeps a single un-chunked
    projection.

    ``kind="cuda"`` requires the ``fast_jl`` CUDA projector, ``"basic"`` forces
    the portable PyTorch one, and ``"auto"`` prefers CUDA and falls back.
    """
    if kind not in PROJECTOR_KINDS:
        raise ValueError(f"Unknown projector kind {kind!r}; choose from {PROJECTOR_KINDS}")

    if kind != "basic":
        unavailable = None
        if not str(device).startswith("cuda"):
            unavailable = f"device {device!r} is not CUDA"
        elif proj_dim % _CUDA_DIM_MULTIPLE:
            unavailable = f"proj_dim={proj_dim} is not a multiple of {_CUDA_DIM_MULTIPLE}"
        else:
            try:
                return CudaProjector(
                    grad_dim=grad_dim,
                    proj_dim=proj_dim,
                    seed=seed,
                    proj_type=ProjectionType.rademacher,
                    device=device,
                    max_batch_size=max_batch_size,
                )
            except (ImportError, RuntimeError, AttributeError) as error:
                unavailable = repr(error)
        if kind == "cuda":
            raise RuntimeError(f"The CUDA projector is unavailable: {unavailable}")
        logger.warning("Using the slower BasicProjector: %s", unavailable)

    return BasicProjector(
        grad_dim=grad_dim,
        proj_dim=proj_dim,
        seed=seed,
        proj_type=ProjectionType.rademacher,
        device=device,
        block_size=min(64, proj_dim),
        dtype=torch.float32,
    )
