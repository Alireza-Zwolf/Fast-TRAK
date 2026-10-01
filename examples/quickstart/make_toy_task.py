"""Generate a small synthetic task with known label noise.

Writes a candidate pool in which a fraction of the labels are deliberately
wrong, plus a clean held-out target set. Because the corrupted rows are known,
the task doubles as a sanity check for attribution: training on a mislabelled
example should hurt clean targets, so those rows should sink to the bottom of
the ranking. Nothing here is real data; every sentence is produced from the
templates below.
"""

from __future__ import annotations

import argparse
import csv
import os
import random

TOPICS = {
    "sports": [
        "the striker",
        "the goalkeeper",
        "the marathon runner",
        "the tennis champion",
        "the coach",
        "the relay team",
        "the referee",
        "the cyclist",
    ],
    "cooking": [
        "the pastry chef",
        "the sourdough starter",
        "the simmering broth",
        "the line cook",
        "the cast-iron skillet",
        "the spice blend",
        "the baker",
        "the marinade",
    ],
    "finance": [
        "the bond trader",
        "the quarterly dividend",
        "the hedge fund",
        "the central bank",
        "the mortgage lender",
        "the stock index",
        "the auditor",
        "the venture investor",
    ],
    "weather": [
        "the cold front",
        "the thunderstorm",
        "the heat wave",
        "the morning fog",
        "the blizzard",
        "the sea breeze",
        "the hailstorm",
        "the monsoon",
    ],
}
TEMPLATES = [
    "A short report about {subject} and what happened next.",
    "Everyone was talking about {subject} this week.",
    "The article explains why {subject} matters so much.",
    "Here is a quick note on {subject} from yesterday.",
    "People keep asking about {subject} lately.",
    "This story centres on {subject} and little else.",
]
LABELS = list(TOPICS)


def generate(rng: random.Random, count: int, templates: list[str]) -> list[tuple[str, str]]:
    rows = []
    for _ in range(count):
        label = rng.choice(LABELS)
        rows.append((rng.choice(templates).format(subject=rng.choice(TOPICS[label])), label))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_candidates", type=int, default=1024)
    parser.add_argument("--num_targets", type=int, default=64)
    parser.add_argument(
        "--noise", type=float, default=0.25, help="fraction of candidate labels to corrupt"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Targets use templates the candidates never see.
    candidates = generate(rng, args.num_candidates, TEMPLATES[:4])
    targets = generate(rng, args.num_targets, TEMPLATES[4:])

    with open(
        os.path.join(args.out_dir, "candidates.tsv"), "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["question", "answer", "is_mislabelled"])
        for question, label in candidates:
            corrupt = rng.random() < args.noise
            if corrupt:
                label = rng.choice([other for other in LABELS if other != label])
            writer.writerow([question, label, int(corrupt)])

    with open(
        os.path.join(args.out_dir, "targets.tsv"), "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["question", "answer"])
        writer.writerows(targets)

    print(f"Wrote {len(candidates)} candidates and {len(targets)} targets to {args.out_dir}")
    print("Labels:", " ".join(LABELS))


if __name__ == "__main__":
    main()
