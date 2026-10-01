# FAST-TRAK

**Exact, batched [TRAK](https://github.com/MadryLab/trak) data attribution for LoRA-tuned causal language models, built for Qwen 3.5.**

TRAK estimates how much each training example helps or hurts a model's
prediction on a held-out example. It needs one gradient *per example*, which
upstream TRAK obtains with `torch.func.vmap` over a functionalised model.
Qwen 3.5 cannot be traced that way: its gated DeltaNet layers use custom
kernels and data-dependent control flow.

FAST-TRAK replaces that step. It recovers the exact per-example gradient of
every LoRA weight from **one ordinary batched forward and backward pass**, then
hands the result to TRAK's unmodified projection and scoring code.

| | |
|---|---|
| **Exact** | Matches per-example autograd to 6e-6 relative error in float32 on Qwen3.5-0.8B |
| **Fast** | 205 examples/s on one L40S, 40x a per-example loop |
| **Drop-in** | Depends on stock `traker`; no fork, no patched Transformers |
| **General** | Works on any PEFT LoRA causal LM; Qwen 3.5 is the tested case |

## Install

```bash
git clone https://github.com/Alireza-Zwolf/FAST-TRAK.git
cd FAST-TRAK
pip install -e .
```

Two optional extras matter on a GPU:

```bash
pip install fast_jl     # TRAK's CUDA random projector (needs a CUDA compiler toolchain)
pip install fla-core    # Triton kernel for Qwen 3.5's gated delta rule (2.4x faster)
```

Without `fast_jl` the portable PyTorch projector is used. Without `fla-core`
Qwen 3.5 runs its slower pure-PyTorch fallback, and a warning says so.

## Quickstart

The quickstart generates a toy topic-classification task, fine-tunes two LoRA
reference adapters, attributes, and checks the result. It needs one CUDA GPU
and takes about two minutes on an L40S.

```bash
bash examples/quickstart/run.sh
```

A quarter of the toy candidates carry a deliberately wrong label, so the
ranking can be checked against ground truth:

```text
candidates: 1024  mislabelled: 245 (24%)
AUROC, clean ranked above mislabelled: 0.805  (0.5 = chance)
mislabelled share of the top 25%:    17%
mislabelled share of the bottom 25%: 69%
```

## Usage

You need LoRA adapters fine-tuned on (subsets of) the candidate pool, a file
of candidates and a file of held-out targets. Data files are headed TSV or CSV
with `question` and `answer` columns.

```bash
ARGS=(--base_model Qwen/Qwen3.5-0.8B-Base
      --checkpoints refs/seed0 refs/seed1 refs/seed2
      --candidates pool.tsv --labels yes no maybe
      --save_dir trak_store --proj_dim 1024 --num_projections 8)

fast-trak setup     "${ARGS[@]}"                  # create the store, once
fast-trak featurize "${ARGS[@]}" --checkpoint 0   # one job per checkpoint, in parallel
fast-trak featurize "${ARGS[@]}" --checkpoint 1
fast-trak featurize "${ARGS[@]}" --checkpoint 2
fast-trak score     "${ARGS[@]}" --targets validation.tsv --exp_name val

fast-trak select --scores trak_store/val_scores.csv --n 5000 --policy top --out top5000.tsv
```

`fast-trak run` chains setup, featurize and score in one process. `score`
writes a ranking, most helpful candidate first:

```text
rank,row,question,answer,score
```

The full `[candidates, targets]` matrix stays in the store at
`trak_store/scores/val.mmap` and loads with `numpy.load(path, mmap_mode="r")`.
A positive entry predicts that training on the candidate raises the target's
correct-label margin.

The same stages are available from Python:

```python
from fast_trak import TrakConfig, featurize, score, setup

config = TrakConfig(
    base_model="Qwen/Qwen3.5-0.8B-Base",
    checkpoints=("refs/seed0", "refs/seed1"),
    candidates_file="pool.tsv",
    labels=("yes", "no", "maybe"),
    save_dir="trak_store",
)
setup(config)
featurize(config)                                   # all checkpoints
scores = score(config, "validation.tsv", exp_name="val")   # [candidates, targets]
```

`BatchedLoRAGradientComputer` can also be passed straight to TRAK's own
`TRAKer(gradient_computer=...)` with a custom model output. See
[docs/design.md](docs/design.md).

## How it works

A LoRA layer is a linear map, and for a linear layer `y = W x` the gradient of
a per-example scalar `f_b` is a sum of outer products over token positions:

```text
d f_b / d W  =  sum_t  (d f_b / d y[b,t])  x[b,t]^T
```

Examples in a batch do not interact, so back-propagating the *sum* of the
per-example outputs leaves each example's own output-gradient in its own batch
row. FAST-TRAK captures `x` with a forward hook and `d f / d y` with a tensor
hook on each LoRA `A` and `B` layer, and one `einsum` per layer yields every
example's exact gradient. Nothing is approximated, and the model runs as
ordinary PyTorch.

## Why it is fast

- **One backward pass per batch.** There is no vmap and no per-example loop.
- **LoRA-sized gradients.** Only adapter weights are differentiated, and the
  projector is built for that dimension. TRAK's auto-configuration sizes it
  from all model parameters and picks a layout that does not fit.
- **Native kernels stay usable.** The model is never traced, so Qwen 3.5 can
  run the FLA Triton kernel that a vmap-based computer has to give up.
- **Only the logits that matter.** `logits_to_keep=1` runs the vocabulary head
  on the final position instead of the whole sequence.
- **Token-budget batching.** Prompts are length-sorted and batched by padded
  tokens, so short prompts run in large batches and long ones in small ones.
- **One gradient, many projections.** Each batch gradient is projected
  `num_projections` times instead of being recomputed per projection.
- **I/O off the hot path.** A checkpoint's memory-maps open once, all
  projections leave the GPU in one transfer, and metadata is written once.
- **Resumable and OOM-safe.** Finished rows are skipped on restart, and a batch
  that runs out of memory is halved, which cannot change the result.

One correctness fix rides along: the output-to-loss factor `1 - p` is computed
in float32. In bfloat16 it rounds to exactly zero once `p` passes about 0.998,
which silently zeroes the score of every confidently fitted training example.

## Results

All numbers below were measured on a single NVIDIA L40S with
Qwen3.5-0.8B-Base, a rank-16 LoRA (5.5M adapter weights) and 2,000 prompts of
up to 512 tokens from the Oral Argument Question Purpose task.

| Configuration | Batched hooks | Per-example loop | Speed-up | Peak GPU |
|---|---|---|---|---|
| bfloat16, FLA kernel | 205 ex/s | 5.2 ex/s | 40x | 11.9 GiB |
| bfloat16, PyTorch fallback | 85 ex/s | 1.5 ex/s | 57x | 24.6 GiB |

Agreement with per-example autograd, each example run alone:

| Setting | Max relative L2 error | Min cosine similarity |
|---|---|---|
| float64, CPU unit tests | below 1e-8 | n/a |
| float32, PyTorch fallback | 6.0e-6 | 0.9999998 |
| float32, FLA kernel | 1.8e-3 | 0.999999 |
| bfloat16, FLA kernel | 8.0e-2 | 0.997 |

The hooks are exact; the larger figures in the lower rows are the kernel's and
bfloat16's own rounding. Reproduce both tables with
[`benchmarks/throughput.py`](benchmarks/throughput.py).

### Validated on a LegalBench task

FAST-TRAK was developed for a thesis on low-resource legal text
classification and checked on the Oral Argument Question Purpose task from
[LegalBench](https://hazyresearch.stanford.edu/legalbench/), using a private
synthetic candidate pool that is not released here.

On Qwen3.5-0.8B, FAST-TRAK reproduces the research implementation it was
extracted from: on a 2,000 x 96 score matrix the two differ by a relative
Frobenius norm of 8.7e-4, the same as two runs of either one (9.4e-4), and
their candidate rankings have a Spearman correlation of 0.999998.

The figure shows what such scores are for. Subsets of a 461k-example
synthetic pool were chosen by score, a student was fine-tuned on each, and
accuracy was measured on real held-out test data.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/oaqp_data_scaling_dark.png">
  <img src="docs/assets/oaqp_data_scaling_light.png" alt="Test accuracy against subset size for TRAK top-N, random and TRAK bottom-N subsets. From 25k examples up, top-N stays between 40 and 47 percent, random near 36 percent and bottom-N between 21 and 28 percent.">
</picture>

Above 10k examples the highest-scored subsets beat random subsets of the same
size by 3 to 10 points, and the lowest-scored ones trail random by 8 to 17
points. Below 10k, random selection is as good or better.

This run used Qwen2.5-0.5B, with scores from the `torch.func` TRAK path that
architecture supports, and predates the hook backend. It will be replaced by
a Qwen 3.5 run scored with FAST-TRAK.

## Scope and limitations

- Only trainable LoRA `A`/`B` `nn.Linear` weights are supported. DoRA, bias
  terms, embeddings and full fine-tuning are rejected with an error rather
  than silently skipped.
- The task is closed-label classification read from the first answer token, so
  labels must start with pairwise-distinct tokens.
- `proj_dim` must stay well below the number of candidates. TRAK inverts a
  `proj_dim x proj_dim` kernel estimated from the candidates, and the scores
  turn to noise as the two approach each other.
- Scores are estimates of a linearised model. Use them to rank candidates, and
  confirm any selection by retraining against a random subset of equal size.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The suite runs on CPU in seconds. It checks the hook gradients against
autograd, the multi-projection store byte-for-byte against stock TRAK, and the
full pipeline against a from-scratch TRAK computation on a tiny LoRA-adapted
Llama built offline.

## Acknowledgements and citation

FAST-TRAK builds on [TRAK](https://github.com/MadryLab/trak) (Park et al.,
2023) and reuses its projectors, savers and score computation unchanged. If
you use this package, please cite the TRAK paper:

```bibtex
@inproceedings{park2023trak,
  title     = {TRAK: Attributing Model Behavior at Scale},
  author    = {Park, Sung Min and Georgiev, Kristian and Ilyas, Andrew and Leclerc, Guillaume and Madry, Aleksander},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2023}
}
```

Released under the [MIT License](LICENSE).
