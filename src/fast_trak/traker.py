"""A TRAKer that shares each batch gradient across several random projections."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import torch
from torch import Tensor
from trak import TRAKer
from trak.savers import ModelIDException
from trak.utils import vectorize

from .store import update_metadata


class MultiProjectionTRAKer(TRAKer):
    """TRAK with ``num_projections`` independent projections per checkpoint.

    Averaging TRAK scores over several random projections reduces projection
    noise. Doing that with the stock ``TRAKer`` means re-running the model once
    per projection; here every batch gradient is computed once and projected
    ``num_projections`` times.

    Each ``(checkpoint, projection)`` pair is stored under a *virtual* model ID
    ``checkpoint * num_projections + projection``, so TRAK's own saver,
    feature finalisation and score aggregation treat every projection as one
    more ensemble member. With ``num_projections=1`` the store is laid out
    exactly as the stock ``TRAKer`` would lay it out.

    I/O is kept off the per-batch path: a checkpoint's memory-maps are opened
    once, all projections leave the device in one transfer, and per-model
    metadata is written once at finalisation instead of once per batch.
    """

    def __init__(
        self,
        *args,
        num_checkpoints: int,
        num_projections: int = 1,
        **kwargs,
    ) -> None:
        if num_projections <= 0 or num_checkpoints <= 0:
            raise ValueError(
                "num_projections and num_checkpoints must be positive, got "
                f"{num_projections} and {num_checkpoints}"
            )
        self.num_projections = int(num_projections)
        self.num_checkpoints = int(num_checkpoints)
        self.current_checkpoint: int | None = None
        self._train_stores: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._target_stores: dict[int, np.ndarray] = {}
        super().__init__(*args, **kwargs)
        self._record_layout()

    # ------------------------------------------------------------ model IDs

    def virtual_ids(self, checkpoint: int) -> list[int]:
        """Virtual model IDs of one checkpoint, one per projection."""
        if not 0 <= checkpoint < self.num_checkpoints:
            raise ValueError(
                f"checkpoint must be in [0, {self.num_checkpoints - 1}], got {checkpoint}"
            )
        first = checkpoint * self.num_projections
        return list(range(first, first + self.num_projections))

    def all_virtual_ids(self) -> list[int]:
        """Every virtual model ID in the ensemble."""
        return list(range(self.num_checkpoints * self.num_projections))

    def _record_layout(self) -> None:
        """Persist the ensemble layout, or check it against an existing store."""
        layout = {
            "num_checkpoints": self.num_checkpoints,
            "num_projections": self.num_projections,
            "projector_seed": self.proj_seed,
        }
        update_metadata(
            self.save_dir / "metadata.json",
            layout,
            has_features=bool(self.saver.model_ids),
        )

    # ----------------------------------------------------------- featurising

    def load_checkpoint(
        self,
        checkpoint: Iterable[Tensor],
        model_id: int,
        _allow_featurizing_already_registered: bool = False,
    ) -> None:
        """Load one checkpoint and open the stores of all its projections."""
        self._train_stores = {}
        for virtual_id in self.virtual_ids(model_id):
            if self.saver.model_ids.get(virtual_id) is None:
                self.saver.register_model_id(virtual_id, _allow_featurizing_already_registered)
            else:
                self.saver.load_current_store(virtual_id)
            store = self.saver.current_store
            self._train_stores[virtual_id] = (
                store["grads"],
                store["out_to_loss"],
                store["is_featurized"],
            )
        self._load_weights(checkpoint)
        self._last_ind = 0
        self.current_checkpoint = model_id
        self.ckpt_loaded = model_id

    def featurize(
        self,
        batch: Iterable[Tensor],
        inds: Iterable[int] | None = None,
        num_samples: int | None = None,
    ) -> int:
        """Featurise one batch for every projection; returns rows written.

        Rows already present in the store are skipped, so an interrupted run
        resumes without recomputation. Nothing is written until all GPU work
        for the batch has succeeded.
        """
        if self.current_checkpoint is None or not self._train_stores:
            raise AssertionError("Call load_checkpoint before featurize")
        batch_inds = self._batch_indices(inds, num_samples, "_last_ind")

        pending = {
            virtual_id: np.flatnonzero(is_featurized[batch_inds].reshape(-1) != 1)
            for virtual_id, (_, _, is_featurized) in self._train_stores.items()
        }
        active_ids = [virtual_id for virtual_id, rows in pending.items() if len(rows)]
        if not active_ids:
            return 0

        raw_grads = self._per_example_grads(batch)
        loss_grads = self.gradient_computer.compute_loss_grad(batch).to(self.dtype).cpu().numpy()
        projected = self._project_all(raw_grads, active_ids)
        del raw_grads

        for column, virtual_id in enumerate(active_ids):
            rows = pending[virtual_id]
            destination = batch_inds[rows]
            grads, out_to_loss, is_featurized = self._train_stores[virtual_id]
            grads[destination] = projected[column][rows]
            out_to_loss[destination] = loss_grads[rows]
            is_featurized[destination] = 1
        return max(len(rows) for rows in pending.values())

    def finalize_features(
        self,
        model_ids: Iterable[int] | None = None,
        del_grads: bool = False,
    ) -> None:
        """Compute TRAK features for the given virtual model IDs (default all)."""
        ids = list(self.saver.model_ids) if model_ids is None else list(model_ids)
        for arrays in self._train_stores.values():
            for array in arrays:
                array.flush()
        # featurize() defers this bookkeeping; settle it before TRAK checks it.
        for virtual_id in ids:
            if self.saver.model_ids.get(virtual_id) is None:
                raise ModelIDException(f"Model ID {virtual_id} is not registered")
            self.saver.load_current_store(virtual_id)
            self.saver.serialize_current_model_id_metadata()
        super().finalize_features(model_ids=ids, del_grads=del_grads)

    # --------------------------------------------------------------- scoring

    def start_scoring_checkpoint(
        self,
        exp_name: str,
        checkpoint: Iterable[Tensor],
        model_id: int,
        num_targets: int,
    ) -> None:
        """Load one checkpoint and open the target stores of its projections."""
        self._target_stores = {}
        for virtual_id in self.virtual_ids(model_id):
            if self.saver.model_ids.get(virtual_id, {}).get("is_finalized") != 1:
                raise ModelIDException(
                    f"Model ID {virtual_id} (checkpoint {model_id}) is not finalized; "
                    "featurize and finalize it before scoring"
                )
            self.saver.init_experiment(exp_name, num_targets, virtual_id)
            self._target_stores[virtual_id] = self.saver.current_store[f"{exp_name}_grads"]
        self._load_weights(checkpoint)
        self._last_ind_target = 0
        self.current_checkpoint = model_id

    def score(
        self,
        batch: Iterable[Tensor],
        inds: Iterable[int] | None = None,
        num_samples: int | None = None,
    ) -> None:
        """Project one batch of target gradients for every projection."""
        if self.current_checkpoint is None or not self._target_stores:
            raise AssertionError("Call start_scoring_checkpoint before score")
        batch_inds = self._batch_indices(inds, num_samples, "_last_ind_target")
        virtual_ids = list(self._target_stores)
        projected = self._project_all(self._per_example_grads(batch), virtual_ids)
        for column, virtual_id in enumerate(virtual_ids):
            self._target_stores[virtual_id][batch_inds] = projected[column]

    def finalize_scores(
        self,
        exp_name: str,
        model_ids: Iterable[int] | None = None,
        allow_skip: bool = False,
    ) -> Tensor:
        """Average scores over the ensemble (default: every checkpoint and projection)."""
        for array in self._target_stores.values():
            array.flush()
        ids = self.all_virtual_ids() if model_ids is None else list(model_ids)
        return super().finalize_scores(exp_name, model_ids=ids, allow_skip=allow_skip)

    # --------------------------------------------------------------- helpers

    def _load_weights(self, checkpoint: Iterable[Tensor]) -> None:
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.gradient_computer.load_model_params(self.model)
        # BasicProjector caches the matrix of the last model ID it projected
        # for, and TRAK frees that matrix when finalising features.
        if hasattr(self.projector, "model_id"):
            self.projector.model_id = None

    def _batch_indices(
        self, inds: Iterable[int] | None, num_samples: int | None, counter: str
    ) -> np.ndarray:
        if (inds is None) == (num_samples is None):
            raise AssertionError("Specify exactly one of inds and num_samples")
        if inds is not None:
            return np.asarray(inds).reshape(-1)
        start = getattr(self, counter)
        setattr(self, counter, start + num_samples)
        return np.arange(start, start + num_samples)

    def _per_example_grads(self, batch: Iterable[Tensor]) -> Tensor:
        grads = self.gradient_computer.compute_per_sample_grad(batch=batch)
        return vectorize(grads, device=self.device) if isinstance(grads, dict) else grads

    def _project_all(self, raw_grads: Tensor, virtual_ids: list[int]) -> np.ndarray:
        """Project with every listed ID's matrix; one device-to-host copy."""
        projected = torch.stack(
            [
                self.projector.project(raw_grads, model_id=virtual_id) / self.normalize_factor
                for virtual_id in virtual_ids
            ]
        )
        return projected.to(self.dtype).cpu().numpy()


def run_with_oom_split(
    step: Callable[..., object],
    indices: np.ndarray,
    batch: tuple[Tensor, ...],
    on_oom: Callable[[], None] | None = None,
) -> int:
    """Run ``step(batch=..., inds=...)``, halving the batch on CUDA OOM.

    Per-example gradients do not depend on batch-mates, so splitting leaves
    results unchanged. ``on_oom`` is called before each retry to release
    cached tensors. Returns the number of splits performed.
    """
    try:
        step(batch=batch, inds=indices)
        return 0
    except torch.cuda.OutOfMemoryError:
        if len(indices) == 1:
            raise
    # Retried outside the except block so the traceback stops pinning tensors.
    if on_oom is not None:
        on_oom()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    half = len(indices) // 2
    splits = 1
    for part in (slice(0, half), slice(half, None)):
        splits += run_with_oom_split(
            step, indices[part], tuple(tensor[part] for tensor in batch), on_oom
        )
    return splits
