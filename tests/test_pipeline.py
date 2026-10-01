"""The whole pipeline on CPU, checked against a from-scratch TRAK computation."""

from __future__ import annotations

import csv
import math

import numpy as np
import pytest
import torch
from trak.projectors import BasicProjector, ProjectionType

from fast_trak import TrakConfig, featurize, open_session, score, setup
from fast_trak.cli import main
from fast_trak.data import LeftPadCollator, PromptDataset, read_examples
from fast_trak.labels import label_first_token_ids
from fast_trak.models import load_model, load_tokenizer
from fast_trak.outputs import AnswerMarginOutput
from fast_trak.selection import read_ranking

PROJ_DIM, NUM_PROJECTIONS, SEED = 8, 2, 3


def make_config(task, save_dir, **overrides):
    settings = dict(
        base_model=task.base_model,
        checkpoints=task.checkpoints,
        candidates_file=task.candidates_file,
        labels=task.labels,
        save_dir=str(save_dir),
        max_length=64,
        proj_dim=PROJ_DIM,
        num_projections=NUM_PROJECTIONS,
        projector_seed=SEED,
        projector="basic",
        model_dtype="float32",
        max_tokens_per_batch=40,
        device="cpu",
    )
    settings.update(overrides)
    return TrakConfig(**settings)


def reference_scores(task):
    """TRAK written out directly: autograd per example, dense linear algebra."""
    tokenizer = load_tokenizer(task.base_model)
    output = AnswerMarginOutput(label_first_token_ids(tokenizer, task.labels))
    collate = LeftPadCollator(tokenizer.pad_token_id)

    def gradients(model, path):
        examples = read_examples(path, task.labels)
        parameters = [p for p in model.parameters() if p.requires_grad]
        rows, factors = [], []
        for sample in PromptDataset(examples, tokenizer, task.labels, 64).samples:
            margin, factor = output.forward_batched(model, collate([sample])[1:])
            grads = torch.autograd.grad(margin.sum(), parameters)
            rows.append(torch.cat([g.reshape(-1) for g in grads]))
            factors.append(factor[0, 0])
        return torch.stack(rows), torch.stack(factors)

    score_sum, factor_sum, members = 0.0, 0.0, 0
    for checkpoint, adapter in enumerate(task.checkpoints):
        model = load_model(task.base_model, adapter, "cpu", "float32")
        train, factors = gradients(model, task.candidates_file)
        targets, _ = gradients(model, task.targets_file)
        grad_dim = train.shape[1]
        for projection in range(NUM_PROJECTIONS):
            projector = BasicProjector(
                grad_dim,
                PROJ_DIM,
                SEED,
                ProjectionType.rademacher,
                "cpu",
                block_size=PROJ_DIM,
                dtype=torch.float32,
            )
            virtual_id = checkpoint * NUM_PROJECTIONS + projection
            phi = projector.project(train, model_id=virtual_id) / math.sqrt(grad_dim)
            phi_targets = projector.project(targets, model_id=virtual_id) / math.sqrt(grad_dim)
            kernel_inverse = torch.linalg.inv(phi.T @ phi)
            kernel_inverse /= kernel_inverse.abs().mean()  # TRAK's per-member rescaling
            score_sum = score_sum + phi @ kernel_inverse @ phi_targets.T
            factor_sum = factor_sum + factors
            members += 1
    return ((score_sum / members) * (factor_sum / members)[:, None]).numpy()


def test_pipeline_matches_a_direct_trak_computation(tiny_task, tmp_path):
    config = make_config(tiny_task, tmp_path / "store")
    setup(config)
    for checkpoint in range(len(config.checkpoints)):  # as separate jobs would
        featurize(config, checkpoint=checkpoint)
    scores = score(config, tiny_task.targets_file, exp_name="val")

    expected = reference_scores(tiny_task)
    assert scores.shape == expected.shape == (24, 5)
    np.testing.assert_allclose(scores, expected, rtol=2e-3, atol=1e-5 * np.abs(expected).max())

    ranking = read_ranking(str(tmp_path / "store" / "val_scores.csv"))
    order = np.argsort(-expected.mean(axis=1), kind="stable")
    assert [int(row["row"]) for row in ranking] == order.tolist()
    assert [float(row["score"]) for row in ranking] == sorted(
        (float(row["score"]) for row in ranking), reverse=True
    )


def test_batching_does_not_change_scores(tiny_task, tmp_path):
    small = make_config(tiny_task, tmp_path / "small", max_tokens_per_batch=1, pad_to_multiple_of=4)
    large = make_config(tiny_task, tmp_path / "large", max_tokens_per_batch=10_000)
    results = []
    for config in (small, large):
        session = open_session(config)
        featurize(config, session=session)
        results.append(score(config, tiny_task.targets_file, session=session))
    np.testing.assert_allclose(results[0], results[1], rtol=2e-3, atol=1e-6)


def test_rerunning_featurize_is_a_no_op(tiny_task, tmp_path):
    config = make_config(tiny_task, tmp_path / "store")
    featurize(config)
    session = open_session(config)
    session.traker.load_checkpoint(session.activate(0), model_id=0)
    written = []
    session.run_batches(
        lambda batch, inds: written.append(session.traker.featurize(batch, inds)),
        session.candidates,
    )
    assert set(written) == {0}


@pytest.mark.parametrize(
    "change",
    [{"max_length": 32}, {"prompt_template": "Q: {question} A: "}, {"num_projections": 1}],
)
def test_a_store_rejects_different_settings(tiny_task, tmp_path, change):
    featurize(make_config(tiny_task, tmp_path / "store"), checkpoint=0)
    with pytest.raises(ValueError, match="different settings"):
        open_session(make_config(tiny_task, tmp_path / "store", **change))


def test_a_store_rejects_a_different_candidate_file(tiny_task, tmp_path):
    featurize(make_config(tiny_task, tmp_path / "store"), checkpoint=0)
    edited = tmp_path / "candidates.tsv"
    lines = open(tiny_task.candidates_file, encoding="utf-8").read().splitlines()
    edited.write_text("\n".join([lines[0], *reversed(lines[1:])]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="candidates_sha256"):
        open_session(make_config(tiny_task, tmp_path / "store", candidates_file=str(edited)))


def test_cli_runs_end_to_end_and_selects_a_subset(tiny_task, tmp_path):
    store, subset = tmp_path / "store", tmp_path / "top.tsv"
    main(
        [
            "run",
            "--base_model",
            tiny_task.base_model,
            "--checkpoints",
            *tiny_task.checkpoints,
            "--candidates",
            tiny_task.candidates_file,
            "--targets",
            tiny_task.targets_file,
            "--labels",
            *tiny_task.labels,
            "--save_dir",
            str(store),
            "--proj_dim",
            str(PROJ_DIM),
            "--projector",
            "basic",
            "--model_dtype",
            "float32",
            "--device",
            "cpu",
        ]
    )
    ranking = read_ranking(str(store / "targets_scores.csv"))
    assert len(ranking) == 24

    main(
        ["select", "--scores", str(store / "targets_scores.csv"), "--n", "5", "--out", str(subset)]
    )
    with open(subset, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [(r["question"], r["answer"]) for r in rows] == [
        (r["question"], r["answer"]) for r in ranking[:5]
    ]
