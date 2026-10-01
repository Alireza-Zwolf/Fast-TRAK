"""The three attribution stages: set up a store, featurise, score.

The stages are separate so they can run as separate jobs. ``setup`` is cheap
and creates the store; ``featurize`` is the expensive part and runs
independently per checkpoint (one GPU job each, safely resumable); ``score``
needs every checkpoint featurised and can be repeated for any number of
target sets against the same features.
"""

from __future__ import annotations

import csv
import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import qwen35
from .data import DEFAULT_PROMPT_TEMPLATE, Example, build_loader, read_examples
from .gradients import BatchedLoRAGradientComputer
from .labels import label_first_token_ids
from .models import (
    load_adapter_weights,
    load_model,
    load_tokenizer,
    model_type,
    trainable_parameter_names,
)
from .outputs import AnswerMarginOutput
from .projection import build_projector
from .store import file_sha256, update_metadata
from .traker import MultiProjectionTRAKer, run_with_oom_split

logger = logging.getLogger(__name__)

STORE_FORMAT = "fast-trak/1"
FALLBACK_TOKEN_BUDGET = 1024


@dataclass(frozen=True)
class TrakConfig:
    """Everything that defines an attribution store.

    Attributes:
        base_model: Hugging Face name or path of the base causal LM.
        checkpoints: LoRA adapter directories, one per ensemble member.
        candidates_file: headed TSV/CSV of training candidates to score.
        labels: the closed label set, in a fixed order.
        save_dir: where the TRAK store lives.
        question_col, answer_col: column names in the data files.
        prompt_template: prompt format; must end where the answer begins.
        max_length: prompts are left-truncated to this many tokens.
        num_candidates: use only the first N candidate rows.
        drop_unknown_labels: skip rows with an out-of-set answer instead of failing.
        proj_dim: dimension of each random projection.
        num_projections: independent projections per checkpoint.
        projector_seed: base seed of the projections.
        projector: ``"auto"``, ``"cuda"`` (fast_jl) or ``"basic"``.
        proj_max_batch_size: fast_jl batch size (8, 16 or 32).
        model_dtype: ``"bfloat16"`` or ``"float32"``.
        max_tokens_per_batch: padded-token budget per batch (default: per model).
        max_batch_size: optional cap on examples per batch.
        pad_to_multiple_of: round batch widths up to this multiple (default: per model).
        fla_kernel: Qwen 3.5 kernel; ``None`` uses it when installed.
        device: torch device.
    """

    base_model: str
    checkpoints: tuple[str, ...]
    candidates_file: str
    labels: tuple[str, ...]
    save_dir: str
    question_col: str = "question"
    answer_col: str = "answer"
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    max_length: int = 512
    num_candidates: int | None = None
    drop_unknown_labels: bool = False
    proj_dim: int = 2048
    num_projections: int = 1
    projector_seed: int = 0
    projector: str = "auto"
    proj_max_batch_size: int = 32
    model_dtype: str = "bfloat16"
    max_tokens_per_batch: int | None = None
    max_batch_size: int | None = None
    pad_to_multiple_of: int | None = None
    fla_kernel: bool | None = None
    device: str = "cuda"

    def __post_init__(self) -> None:
        if not self.checkpoints:
            raise ValueError("At least one checkpoint is required")
        if len(set(self.labels)) != len(self.labels) or not self.labels:
            raise ValueError("labels must be non-empty and distinct")
        for directory in self.checkpoints:
            if not os.path.isdir(directory):
                raise FileNotFoundError(f"Checkpoint directory not found: {directory}")


@dataclass
class Session:
    """A loaded model and its open store, ready to featurise or score."""

    config: TrakConfig
    tokenizer: object
    model: torch.nn.Module
    traker: MultiProjectionTRAKer
    candidates: list[Example]
    max_tokens_per_batch: int
    pad_to_multiple_of: int | None

    def loader(self, examples: Sequence[Example]):
        return build_loader(
            examples,
            self.tokenizer,
            self.config.labels,
            max_length=self.config.max_length,
            max_tokens_per_batch=self.max_tokens_per_batch,
            max_batch_size=self.config.max_batch_size,
            pad_to_multiple_of=self.pad_to_multiple_of,
            prompt_template=self.config.prompt_template,
        )

    def read(self, path: str, limit: int | None = None) -> list[Example]:
        return read_examples(
            path,
            self.config.labels,
            question_col=self.config.question_col,
            answer_col=self.config.answer_col,
            drop_unknown_labels=self.config.drop_unknown_labels,
            limit=limit,
        )

    def activate(self, checkpoint: int) -> dict:
        """Swap in one checkpoint's adapter and return the model state."""
        load_adapter_weights(self.model, self.config.checkpoints[checkpoint], self.config.device)
        return self.model.state_dict()

    def run_batches(self, step, examples: Sequence[Example]) -> tuple[int, int]:
        """Apply ``step`` to every batch; returns ``(batches, OOM splits)``."""
        loader = self.loader(examples)
        splits = 0
        for indices, *tensors in loader:
            batch = tuple(tensor.to(self.config.device) for tensor in tensors)
            splits += run_with_oom_split(
                step, indices.numpy(), batch, on_oom=self.traker.gradient_computer.reset
            )
        return len(loader), splits


def _batching_defaults(config: TrakConfig) -> tuple[int, int | None]:
    """Resolve the token budget and padding multiple for the base model."""
    is_qwen35 = model_type(config.base_model) in qwen35.MODEL_TYPES
    budget = config.max_tokens_per_batch
    if budget is None:
        budget = (qwen35.token_budget(config.base_model) if is_qwen35 else None) or (
            FALLBACK_TOKEN_BUDGET
        )
    padding = config.pad_to_multiple_of
    if padding is None and is_qwen35:
        padding = qwen35.PAD_TO_MULTIPLE_OF
    return budget, padding


def open_session(config: TrakConfig) -> Session:
    """Load the model, open (or create) the store and validate its settings."""
    os.makedirs(config.save_dir, exist_ok=True)
    tokenizer = load_tokenizer(config.base_model)
    label_ids = label_first_token_ids(tokenizer, config.labels, config.prompt_template)
    max_tokens_per_batch, pad_to_multiple_of = _batching_defaults(config)

    candidates = read_examples(
        config.candidates_file,
        config.labels,
        question_col=config.question_col,
        answer_col=config.answer_col,
        drop_unknown_labels=config.drop_unknown_labels,
        limit=config.num_candidates,
    )
    model = load_model(
        config.base_model,
        config.checkpoints[0],
        config.device,
        config.model_dtype,
        config.fla_kernel,
    )
    grad_wrt = trainable_parameter_names(model)
    grad_dim = sum(p.numel() for p in model.parameters() if p.requires_grad)
    projector = build_projector(
        grad_dim,
        config.proj_dim,
        config.projector_seed,
        config.device,
        kind=config.projector,
        max_batch_size=config.proj_max_batch_size,
    )
    traker = MultiProjectionTRAKer(
        model=model,
        task=AnswerMarginOutput(label_ids),
        train_set_size=len(candidates),
        save_dir=config.save_dir,
        device=config.device,
        gradient_computer=BatchedLoRAGradientComputer,
        projector=projector,
        proj_dim=config.proj_dim,
        projector_seed=config.projector_seed,
        grad_wrt=grad_wrt,
        use_half_precision=False,
        logging_level=logging.WARNING,
        num_checkpoints=len(config.checkpoints),
        num_projections=config.num_projections,
    )
    update_metadata(
        Path(config.save_dir) / "metadata.json",
        {
            "store_format": STORE_FORMAT,
            "base_model": config.base_model,
            "model_dtype": config.model_dtype,
            "labels": list(config.labels),
            "prompt_template": config.prompt_template,
            "max_length": config.max_length,
            "pad_to_multiple_of": pad_to_multiple_of,
            "grad_dim": grad_dim,
            "candidates_sha256": file_sha256(config.candidates_file),
        },
        has_features=bool(traker.saver.model_ids),
    )
    logger.info(
        "Opened store %s: %d candidates, %d checkpoint(s) x %d projection(s), "
        "LoRA grad_dim=%s, token budget=%d, padding multiple=%s",
        config.save_dir,
        len(candidates),
        len(config.checkpoints),
        config.num_projections,
        f"{grad_dim:,}",
        max_tokens_per_batch,
        pad_to_multiple_of,
    )
    return Session(
        config=config,
        tokenizer=tokenizer,
        model=model,
        traker=traker,
        candidates=candidates,
        max_tokens_per_batch=max_tokens_per_batch,
        pad_to_multiple_of=pad_to_multiple_of,
    )


def setup(config: TrakConfig) -> None:
    """Create the store and its metadata without computing anything.

    Run this once before launching ``featurize`` jobs in parallel, so that no
    two jobs race to create the store.
    """
    open_session(config)


def featurize(
    config: TrakConfig,
    checkpoint: int | None = None,
    session: Session | None = None,
) -> None:
    """Featurise the candidates for one checkpoint, or all when unspecified."""
    session = session or open_session(config)
    selected = range(len(config.checkpoints)) if checkpoint is None else [checkpoint]
    for index in selected:
        traker = session.traker
        virtual_ids = traker.virtual_ids(index)
        traker.load_checkpoint(session.activate(index), model_id=index)
        _reset_peak_memory(config.device)
        started = time.perf_counter()
        batches, splits = session.run_batches(traker.featurize, session.candidates)
        _synchronize(config.device)
        elapsed = time.perf_counter() - started
        traker.finalize_features(model_ids=virtual_ids)
        logger.info(
            "Checkpoint %d: %d candidates in %.1fs (%.1f/s, %d batches, %d OOM splits%s)",
            index,
            len(session.candidates),
            elapsed,
            len(session.candidates) / max(elapsed, 1e-9),
            batches,
            splits,
            _peak_memory(config.device),
        )


def score(
    config: TrakConfig,
    targets_file: str,
    exp_name: str = "targets",
    out_csv: str | None = None,
    num_targets: int | None = None,
    session: Session | None = None,
) -> np.ndarray:
    """Score every candidate against the targets; returns ``[candidates, targets]``.

    A positive entry predicts that training on the candidate raises the
    target's correct-label margin. Candidates are ranked by their mean score
    over targets and written to ``out_csv`` (default
    ``<save_dir>/<exp_name>_scores.csv``). The full matrix is kept in the
    store at ``<save_dir>/scores/<exp_name>.mmap`` (NumPy ``.npy`` format).
    """
    session = session or open_session(config)
    traker = session.traker
    targets = session.read(targets_file, limit=num_targets)

    for index in range(len(config.checkpoints)):
        traker.start_scoring_checkpoint(
            exp_name=exp_name,
            checkpoint=session.activate(index),
            model_id=index,
            num_targets=len(targets),
        )
        session.run_batches(traker.score, targets)

    scores = np.asarray(traker.finalize_scores(exp_name=exp_name))
    out_csv = out_csv or os.path.join(config.save_dir, f"{exp_name}_scores.csv")
    write_ranking(out_csv, session.candidates, scores.mean(axis=1))
    logger.info("Scored %d candidates against %d targets -> %s", *scores.shape, out_csv)
    return scores


def write_ranking(path: str, candidates: Sequence[Example], mean_scores: np.ndarray) -> None:
    """Write candidates sorted by descending score, most helpful first."""
    if mean_scores.shape != (len(candidates),):
        raise ValueError(f"{len(candidates)} candidates but scores of shape {mean_scores.shape}")
    order = np.argsort(-mean_scores, kind="stable")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "row", "question", "answer", "score"])
        for rank, position in enumerate(order):
            example = candidates[position]
            writer.writerow(
                [rank, example.row, example.question, example.answer, float(mean_scores[position])]
            )


def _is_cuda(device: str) -> bool:
    return str(device).startswith("cuda")


def _reset_peak_memory(device: str) -> None:
    if _is_cuda(device):
        torch.cuda.reset_peak_memory_stats()


def _synchronize(device: str) -> None:
    if _is_cuda(device):
        torch.cuda.synchronize()


def _peak_memory(device: str) -> str:
    if not _is_cuda(device):
        return ""
    return f", peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB"
