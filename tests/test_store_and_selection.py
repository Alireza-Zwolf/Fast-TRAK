from __future__ import annotations

import json

import pytest

from fast_trak.selection import select
from fast_trak.store import update_metadata


def test_metadata_is_recorded_then_verified(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text('{"JL dimension": 8}', encoding="utf-8")

    update_metadata(path, {"base_model": "m", "labels": ["a", "b"]}, has_features=False)
    assert json.loads(path.read_text()) == {
        "JL dimension": 8,
        "base_model": "m",
        "labels": ["a", "b"],
    }
    update_metadata(path, {"base_model": "m"}, has_features=True)  # unchanged: accepted
    with pytest.raises(ValueError, match="different settings"):
        update_metadata(path, {"base_model": "other"}, has_features=True)


def test_a_populated_store_from_other_code_is_rejected(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text('{"JL dimension": 8}', encoding="utf-8")
    with pytest.raises(ValueError, match="already holds features"):
        update_metadata(path, {"base_model": "m"}, has_features=True)


RANKING = [{"question": f"q{i}", "answer": "a", "score": str(-i)} for i in range(10)]


def test_top_and_bottom_take_the_ends_of_the_ranking():
    assert select(RANKING, 3, "top") == RANKING[:3]
    assert select(RANKING, 3, "bottom") == RANKING[-3:]


def test_random_is_seeded_and_budget_matched():
    first, again = select(RANKING, 4, "random", seed=1), select(RANKING, 4, "random", seed=1)
    assert first == again and len({row["question"] for row in first}) == 4
    assert first != select(RANKING, 4, "random", seed=2)


@pytest.mark.parametrize("n", [0, 11])
def test_rejects_an_impossible_budget(n):
    with pytest.raises(ValueError, match="n must be"):
        select(RANKING, n)
