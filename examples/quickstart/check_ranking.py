"""Check that attribution pushes the known-mislabelled candidates to the bottom.

Reads the ranking written by ``fast-trak score`` and the ``is_mislabelled``
column of the toy candidate file, then reports how well the scores separate
clean rows from corrupted ones.
"""

from __future__ import annotations

import argparse
import csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, help="ranking CSV from `fast-trak score`")
    parser.add_argument("--candidates", required=True, help="candidates.tsv from make_toy_task.py")
    args = parser.parse_args()

    with open(args.candidates, newline="", encoding="utf-8") as handle:
        mislabelled = [int(row["is_mislabelled"]) for row in csv.DictReader(handle, delimiter="\t")]
    with open(args.scores, newline="", encoding="utf-8") as handle:
        ranked = [mislabelled[int(row["row"])] for row in csv.DictReader(handle)]  # best first

    noisy, clean = sum(ranked), len(ranked) - sum(ranked)
    # AUROC: probability that a clean row outranks a mislabelled one.
    clean_seen, pairs_won = 0, 0
    for flag in ranked:
        if flag:
            pairs_won += clean_seen
        else:
            clean_seen += 1
    auroc = pairs_won / (noisy * clean)

    quarter = len(ranked) // 4
    print(f"candidates: {len(ranked)}  mislabelled: {noisy} ({noisy / len(ranked):.0%})")
    print(f"AUROC, clean ranked above mislabelled: {auroc:.3f}  (0.5 = chance)")
    print(f"mislabelled share of the top 25%:    {sum(ranked[:quarter]) / quarter:.0%}")
    print(f"mislabelled share of the bottom 25%: {sum(ranked[-quarter:]) / quarter:.0%}")


if __name__ == "__main__":
    main()
