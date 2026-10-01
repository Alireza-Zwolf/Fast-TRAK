"""Turning a score ranking into a training subset."""

from __future__ import annotations

import csv
import os
import random

POLICIES = ("top", "bottom", "random")


def read_ranking(path: str) -> list[dict[str, str]]:
    """Read a ranking CSV written by :func:`fast_trak.pipeline.write_ranking`."""
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} contains no ranked candidates")
    return rows


def select(
    ranking: list[dict[str, str]], n: int, policy: str = "top", seed: int = 0
) -> list[dict[str, str]]:
    """Choose ``n`` candidates from a ranking sorted best-first.

    ``top`` keeps the highest-scoring candidates, ``bottom`` the lowest, and
    ``random`` a uniform sample: the budget-matched control that shows whether
    the scores add anything over simply having ``n`` examples.
    """
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy {policy!r}; choose from {POLICIES}")
    if not 0 < n <= len(ranking):
        raise ValueError(f"n must be in [1, {len(ranking)}], got {n}")
    if policy == "top":
        return ranking[:n]
    if policy == "bottom":
        return ranking[-n:]
    return random.Random(seed).sample(ranking, n)


def write_subset(path: str, rows: list[dict[str, str]]) -> None:
    """Write the chosen candidates as a ``question``/``answer`` TSV."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["question", "answer"])
        for row in rows:
            writer.writerow([row["question"], row["answer"]])
