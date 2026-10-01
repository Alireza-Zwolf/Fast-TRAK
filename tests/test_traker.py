"""MultiProjectionTRAKer against the stock TRAKer, on a tiny linear problem."""

from __future__ import annotations

import filecmp
from pathlib import Path

import numpy as np
import pytest
import torch
from trak import TRAKer
from trak.projectors import AbstractProjector, ProjectionType
from trak.savers import ModelIDException

from fast_trak.traker import MultiProjectionTRAKer, run_with_oom_split

CANDIDATES = torch.tensor(
    [
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
        [2.0, 1.0, 1.0],
        [0.5, -1.0, 2.0],
        [-1.0, 0.5, 0.0],
    ]
)
TARGETS = torch.tensor([[1.0, 2.0, 0.0], [0.0, 1.0, 2.0], [1.0, -1.0, 1.0]])
CANDIDATE_BATCHES = [np.array([4, 1]), np.array([0, 5, 2]), np.array([3])]
TARGET_BATCHES = [np.array([2]), np.array([0, 1])]
NUM_CHECKPOINTS = 2


class IdentityGradientComputer:
    """Treats each input row as its own gradient and counts backward passes."""

    def __init__(self, model, task, grad_dim, dtype, device, **kwargs):
        self.model = model
        self.per_sample_calls = 0

    def load_model_params(self, model):
        self.model = model

    def compute_per_sample_grad(self, batch):
        self.per_sample_calls += 1
        return batch[0]

    def compute_loss_grad(self, batch):
        value = torch.sigmoid(self.model.weight.detach().reshape(-1)[0])
        return value.expand(batch[0].shape[0], 1)

    def reset(self):
        pass


class SeededProjector(AbstractProjector):
    """A dense Gaussian projection that depends on the model ID, like TRAK's."""

    def __init__(self, seed=17):
        super().__init__(3, 2, seed, ProjectionType.normal, "cpu")

    def project(self, grads, model_id):
        generator = torch.Generator().manual_seed(self.seed + 10_000 * model_id)
        return grads @ torch.randn(
            self.grad_dim, self.proj_dim, generator=generator, dtype=grads.dtype
        )

    def free_memory(self):
        pass


def traker_kwargs(save_dir):
    return dict(
        model=torch.nn.Linear(3, 1, bias=False),
        task=object(),
        train_set_size=len(CANDIDATES),
        save_dir=str(save_dir),
        device="cpu",
        gradient_computer=IdentityGradientComputer,
        projector=SeededProjector(),
        use_half_precision=False,
    )


def make_traker(save_dir, num_projections):
    kwargs = traker_kwargs(save_dir)
    traker = MultiProjectionTRAKer(
        **kwargs, num_checkpoints=NUM_CHECKPOINTS, num_projections=num_projections
    )
    return traker, kwargs["model"]


def weights(model, checkpoint):
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.3 + checkpoint, -0.2, 0.1 * checkpoint]]))
    return model.state_dict()


def run(save_dir, num_projections, featurize_step=None):
    traker, model = make_traker(save_dir, num_projections)
    for checkpoint in range(NUM_CHECKPOINTS):
        traker.load_checkpoint(weights(model, checkpoint), model_id=checkpoint)
        for inds in CANDIDATE_BATCHES:
            (featurize_step or traker.featurize)(batch=(CANDIDATES[inds],), inds=inds)
        traker.finalize_features(model_ids=traker.virtual_ids(checkpoint))
    for checkpoint in range(NUM_CHECKPOINTS):
        traker.start_scoring_checkpoint("val", weights(model, checkpoint), checkpoint, len(TARGETS))
        for inds in TARGET_BATCHES:
            traker.score(batch=(TARGETS[inds],), inds=inds)
    return np.array(traker.finalize_scores("val")), traker


def run_stock_traker(save_dir, num_projections):
    """The same ensemble on upstream TRAK: one full pass per virtual model ID."""
    kwargs = traker_kwargs(save_dir)
    traker, model = TRAKer(**kwargs), kwargs["model"]
    ids = [
        (c, c * num_projections + p) for c in range(NUM_CHECKPOINTS) for p in range(num_projections)
    ]
    for checkpoint, virtual_id in ids:
        traker.load_checkpoint(weights(model, checkpoint), model_id=virtual_id)
        for inds in CANDIDATE_BATCHES:
            traker.featurize(batch=(CANDIDATES[inds],), inds=inds)
    traker.finalize_features()
    for checkpoint, virtual_id in ids:
        traker.start_scoring_checkpoint("val", weights(model, checkpoint), virtual_id, len(TARGETS))
        for inds in TARGET_BATCHES:
            traker.score(batch=(TARGETS[inds],), inds=inds)
    return np.array(traker.finalize_scores("val")), traker


def array_files(root: Path):
    return sorted(path.relative_to(root) for path in root.rglob("*.mmap"))


@pytest.mark.parametrize("num_projections", [1, 3])
def test_store_and_scores_are_identical_to_stock_trak(tmp_path, num_projections):
    expected_scores, stock = run_stock_traker(tmp_path / "stock", num_projections)
    scores, traker = run(tmp_path / "fast", num_projections)

    np.testing.assert_array_equal(scores, expected_scores)
    files = array_files(tmp_path / "stock")
    assert files and files == array_files(tmp_path / "fast")
    for path in files:
        assert filecmp.cmp(tmp_path / "stock" / path, tmp_path / "fast" / path, shallow=False), path

    # Same result, but one backward pass per batch instead of one per projection.
    batches = NUM_CHECKPOINTS * (len(CANDIDATE_BATCHES) + len(TARGET_BATCHES))
    assert traker.gradient_computer.per_sample_calls == batches
    assert stock.gradient_computer.per_sample_calls == batches * num_projections


def test_featurize_resumes_without_recomputing(tmp_path):
    traker, model = make_traker(tmp_path, 3)
    traker.load_checkpoint(weights(model, 0), model_id=0)
    assert (
        traker.featurize(batch=(CANDIDATES[CANDIDATE_BATCHES[0]],), inds=CANDIDATE_BATCHES[0]) == 2
    )
    del traker

    resumed, model = make_traker(tmp_path, 3)
    resumed.load_checkpoint(weights(model, 0), model_id=0)
    written = [
        resumed.featurize(batch=(CANDIDATES[inds],), inds=inds) for inds in CANDIDATE_BATCHES
    ]
    assert written == [0, 3, 1]
    assert resumed.gradient_computer.per_sample_calls == len(CANDIDATE_BATCHES) - 1
    resumed.finalize_features(model_ids=resumed.virtual_ids(0))
    assert all(resumed.saver.model_ids[i]["is_finalized"] == 1 for i in resumed.virtual_ids(0))


def test_finalize_rejects_incomplete_featurization(tmp_path):
    traker, model = make_traker(tmp_path, 2)
    traker.load_checkpoint(weights(model, 0), model_id=0)
    traker.featurize(batch=(CANDIDATES[CANDIDATE_BATCHES[0]],), inds=CANDIDATE_BATCHES[0])
    with pytest.raises(ModelIDException, match="not fully featurized"):
        traker.finalize_features(model_ids=traker.virtual_ids(0))


def test_scoring_requires_finalized_features(tmp_path):
    traker, model = make_traker(tmp_path, 2)
    with pytest.raises(ModelIDException, match="not finalized"):
        traker.start_scoring_checkpoint("val", weights(model, 0), 0, len(TARGETS))


def test_store_layout_cannot_change(tmp_path):
    run(tmp_path, 3)
    with pytest.raises(ValueError, match="num_projections"):
        make_traker(tmp_path, 2)


def test_oom_split_writes_every_row_once(tmp_path):
    expected, _ = run(tmp_path / "reference", 3)
    calls = []

    def run_split(save_dir):
        traker_holder = {}

        def flaky(batch, inds):
            calls.append(len(inds))
            if len(inds) > 1:
                raise torch.cuda.OutOfMemoryError("simulated")
            return traker_holder["traker"].featurize(batch=batch, inds=inds)

        traker, model = make_traker(save_dir, 3)
        traker_holder["traker"] = traker
        for checkpoint in range(NUM_CHECKPOINTS):
            traker.load_checkpoint(weights(model, checkpoint), model_id=checkpoint)
            splits = run_with_oom_split(flaky, np.arange(len(CANDIDATES)), (CANDIDATES,))
            assert splits == len(CANDIDATES) - 1
            traker.finalize_features(model_ids=traker.virtual_ids(checkpoint))
        for checkpoint in range(NUM_CHECKPOINTS):
            traker.start_scoring_checkpoint(
                "val", weights(model, checkpoint), checkpoint, len(TARGETS)
            )
            run_with_oom_split(traker.score, np.arange(len(TARGETS)), (TARGETS,))
        return np.array(traker.finalize_scores("val"))

    np.testing.assert_allclose(run_split(tmp_path / "split"), expected, rtol=1e-6, atol=1e-7)
    assert max(calls) == len(CANDIDATES) and calls.count(1) == NUM_CHECKPOINTS * len(CANDIDATES)
