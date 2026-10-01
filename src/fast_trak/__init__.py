"""FAST-TRAK: exact batched TRAK attribution for LoRA-tuned causal LMs."""

from .gradients import BatchedLoRAGradientComputer
from .outputs import AnswerMarginOutput, BatchedModelOutput
from .pipeline import TrakConfig, featurize, open_session, score, setup
from .traker import MultiProjectionTRAKer

__version__ = "0.1.0"

__all__ = [
    "AnswerMarginOutput",
    "BatchedLoRAGradientComputer",
    "BatchedModelOutput",
    "MultiProjectionTRAKer",
    "TrakConfig",
    "featurize",
    "open_session",
    "score",
    "setup",
]
