#!/usr/bin/env bash
# End-to-end quickstart on a generated toy task. Needs one CUDA GPU.
#
#   bash examples/quickstart/run.sh
#
# Override the model or the output directory with BASE_MODEL=... OUT=...
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-0.8B-Base}"
OUT="${OUT:-./quickstart_out}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LABELS=(sports cooking finance weather)
SEEDS=(0 1)

# 1. A candidate pool with 25% deliberately wrong labels, and clean targets.
python "$HERE/make_toy_task.py" --out_dir "$OUT/data"

# 2. Reference adapters: one LoRA fine-tune of the pool per seed.
CHECKPOINTS=()
for seed in "${SEEDS[@]}"; do
    python "$HERE/train_reference.py" --base_model "$BASE_MODEL" \
        --train_file "$OUT/data/candidates.tsv" --labels "${LABELS[@]}" \
        --out_dir "$OUT/reference_seed$seed" --seed "$seed"
    CHECKPOINTS+=("$OUT/reference_seed$seed")
done

# 3. Attribution. `run` chains setup -> featurize -> score in one process; on a
#    cluster, run `featurize --checkpoint i` as one job per checkpoint instead.
#    TRAK inverts a proj_dim x proj_dim kernel estimated from the candidates, so
#    proj_dim must stay far below the pool size: 32 here for 1,024 candidates.
#    Real pools use the CUDA projector with --proj_dim 1024 or more.
fast-trak run --base_model "$BASE_MODEL" --checkpoints "${CHECKPOINTS[@]}" \
    --candidates "$OUT/data/candidates.tsv" --targets "$OUT/data/targets.tsv" \
    --labels "${LABELS[@]}" --save_dir "$OUT/trak" \
    --projector basic --proj_dim 32 --num_projections 4

# 4. The mislabelled candidates should rank at the bottom.
python "$HERE/check_ranking.py" --scores "$OUT/trak/targets_scores.csv" \
    --candidates "$OUT/data/candidates.tsv"

# 5. Keep the 512 most helpful candidates as a training subset.
fast-trak select --scores "$OUT/trak/targets_scores.csv" --n 512 --out "$OUT/top512.tsv"
echo "Wrote $OUT/top512.tsv"
