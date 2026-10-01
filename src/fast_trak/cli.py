"""Command-line interface: ``fast-trak {setup,featurize,score,run,select}``."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from .data import DEFAULT_PROMPT_TEMPLATE
from .projection import PROJECTOR_KINDS
from .selection import POLICIES


def _store_arguments() -> argparse.ArgumentParser:
    """Arguments that define a store; shared by every attribution stage."""
    parser = argparse.ArgumentParser(add_help=False)
    data = parser.add_argument_group("model and data")
    data.add_argument(
        "--base_model", required=True, help="Hugging Face name or path of the base model"
    )
    data.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        metavar="DIR",
        help="LoRA adapter directories, one per ensemble member",
    )
    data.add_argument(
        "--candidates",
        required=True,
        metavar="FILE",
        help="headed TSV/CSV of training candidates to score",
    )
    data.add_argument("--labels", nargs="+", required=True, help="the closed label set")
    data.add_argument("--save_dir", required=True, help="directory of the TRAK store")
    data.add_argument("--question_col", default="question")
    data.add_argument("--answer_col", default="answer")
    data.add_argument(
        "--prompt_template",
        default=DEFAULT_PROMPT_TEMPLATE,
        help="prompt format with a {question} placeholder, ending where the answer begins",
    )
    data.add_argument(
        "--max_length", type=int, default=512, help="prompts are left-truncated to this many tokens"
    )
    data.add_argument(
        "--num_candidates", type=int, default=None, help="use only the first N candidate rows"
    )
    data.add_argument(
        "--drop_unknown_labels",
        action="store_true",
        help="skip rows whose answer is outside --labels instead of failing",
    )

    projection = parser.add_argument_group("projection")
    projection.add_argument("--proj_dim", type=int, default=2048)
    projection.add_argument(
        "--num_projections",
        type=int,
        default=1,
        help="independent random projections per checkpoint",
    )
    projection.add_argument("--projector_seed", type=int, default=0)
    projection.add_argument("--projector", choices=PROJECTOR_KINDS, default="auto")
    projection.add_argument("--proj_max_batch_size", type=int, default=32, choices=[8, 16, 32])

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--model_dtype", choices=["bfloat16", "float32"], default="bfloat16")
    runtime.add_argument(
        "--max_tokens_per_batch",
        type=int,
        default=None,
        help="padded-token budget per batch (default: chosen per model)",
    )
    runtime.add_argument(
        "--max_batch_size", type=int, default=None, help="optional cap on examples per batch"
    )
    runtime.add_argument(
        "--pad_to_multiple_of",
        type=int,
        default=None,
        help="round batch widths up to this multiple (default: chosen per model)",
    )
    runtime.add_argument(
        "--fla_kernel",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Qwen 3.5 FLA Triton kernel (default: use it when installed)",
    )
    runtime.add_argument("--device", default="cuda")
    return parser


def _target_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    targets = parser.add_argument_group("targets")
    targets.add_argument(
        "--targets",
        required=True,
        metavar="FILE",
        help="headed TSV/CSV of held-out examples to attribute",
    )
    targets.add_argument(
        "--exp_name", default="targets", help="name of this target set inside the store"
    )
    targets.add_argument(
        "--out_csv", default=None, help="ranking output (default: <save_dir>/<exp_name>_scores.csv)"
    )
    targets.add_argument(
        "--num_targets", type=int, default=None, help="use only the first N target rows"
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fast-trak",
        description="Exact batched TRAK data attribution for LoRA-tuned causal language models.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    store, targets = _store_arguments(), _target_arguments()

    commands.add_parser(
        "setup", parents=[store], help="create the store (run once before parallel jobs)"
    )
    featurize = commands.add_parser(
        "featurize", parents=[store], help="featurise candidates for a checkpoint"
    )
    featurize.add_argument(
        "--checkpoint",
        type=int,
        default=None,
        metavar="INDEX",
        help="index into --checkpoints (default: all, sequentially)",
    )
    commands.add_parser(
        "score", parents=[store, targets], help="score candidates against a target set"
    )
    commands.add_parser(
        "run", parents=[store, targets], help="setup, featurize and score in one process"
    )

    select = commands.add_parser("select", help="write a top/bottom/random subset of a ranking")
    select.add_argument("--scores", required=True, metavar="FILE", help="ranking CSV from `score`")
    select.add_argument("--n", type=int, required=True, help="number of candidates to keep")
    select.add_argument("--policy", choices=POLICIES, default="top")
    select.add_argument("--seed", type=int, default=0, help="seed of the random policy")
    select.add_argument("--out", required=True, metavar="FILE", help="output TSV")
    return parser


def _config(args: argparse.Namespace):
    from .pipeline import TrakConfig

    return TrakConfig(
        base_model=args.base_model,
        checkpoints=tuple(args.checkpoints),
        candidates_file=args.candidates,
        labels=tuple(args.labels),
        save_dir=args.save_dir,
        question_col=args.question_col,
        answer_col=args.answer_col,
        prompt_template=args.prompt_template,
        max_length=args.max_length,
        num_candidates=args.num_candidates,
        drop_unknown_labels=args.drop_unknown_labels,
        proj_dim=args.proj_dim,
        num_projections=args.num_projections,
        projector_seed=args.projector_seed,
        projector=args.projector,
        proj_max_batch_size=args.proj_max_batch_size,
        model_dtype=args.model_dtype,
        max_tokens_per_batch=args.max_tokens_per_batch,
        max_batch_size=args.max_batch_size,
        pad_to_multiple_of=args.pad_to_multiple_of,
        fla_kernel=args.fla_kernel,
        device=args.device,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("fast_trak").setLevel(logging.INFO)

    if args.command == "select":
        from . import selection

        rows = selection.select(selection.read_ranking(args.scores), args.n, args.policy, args.seed)
        selection.write_subset(args.out, rows)
        return

    # Imported late so `fast-trak select` and `--help` stay instant.
    from . import pipeline

    config = _config(args)
    if args.command == "setup":
        pipeline.setup(config)
    elif args.command == "featurize":
        pipeline.featurize(config, checkpoint=args.checkpoint)
    else:
        session = pipeline.open_session(config)
        if args.command == "run":
            pipeline.featurize(config, session=session)
        pipeline.score(
            config,
            args.targets,
            exp_name=args.exp_name,
            out_csv=args.out_csv,
            num_targets=args.num_targets,
            session=session,
        )


if __name__ == "__main__":
    main()
