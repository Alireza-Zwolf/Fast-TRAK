# FAST-TRAK

FAST-TRAK tells you **which training examples helped, and which hurt**, a
fine-tuned language model's prediction on a given test example. It works on
LoRA fine-tuned models, including Qwen 3.5.

## What problem does it solve?

*Data attribution* answers the question "which training examples is this
prediction based on?". It gives every training example a score for every test
example: positive if training on it pushed the model towards the right answer,
negative if it pushed the model away. The scores are useful for finding
mislabelled or harmful training data, and for picking the most useful subset
of a large training pool.

[TRAK](https://github.com/MadryLab/trak) (Park et al., 2023) is a widely used
method for computing these scores without retraining the model. It needs the
model's gradient for each training example separately, and its reference
implementation gets them with a PyTorch feature (`torch.func.vmap`) that does
not work on Qwen 3.5.

FAST-TRAK computes the same per-example gradients a different way, in one
ordinary forward and backward pass per batch, and passes them to TRAK's
scoring code unchanged. The result is the same scores, on models TRAK could
not handle, at batch speed.

## Install

```bash
git clone https://github.com/Alireza-Zwolf/Fast-TRAK.git
cd Fast-TRAK
pip install -e .
```

Optional, for speed on a GPU:

```bash
pip install fast_jl     # GPU random projection, recommended for large training pools
pip install fla-core    # faster Qwen 3.5 kernel (2.4x)
```

## Quickstart

One script runs the whole thing on the public AG News topic benchmark. It
fine-tunes two small adapters on 4,000 news articles, then scores each of
them against 100 test articles. It needs one CUDA GPU and takes about two
minutes on an L40S.

```bash
python examples/quickstart.py
```

It prints the training articles that most helped and most hurt one test
article:

```text
Top-10 training articles that share the test article's topic: 52%
(picking training articles at random would give 25%)

Test article [Business]: Allianz to fight US court ruling on WTC attacks MUNICH - German insurance concern ...

Most helpful training articles:
  +0.2105  [Business] Governor Promises Help Reopening Schools Affected By Charley ...
  +0.1414  [Business] Final Round in Cable-ISP Fight WASHINGTON -- The US Supreme Court ...
  +0.1395  [Business] More men charging harassment NEW YORK (CNN/Money) ...

Most harmful training articles:
  -0.2584  [World] Hong Kong bank crushes more than customer's spirits ...
  -0.2047  [World] Experts Examine China Aviation Oil Books (AP) ...
  -0.1964  [World] Harmony Issues Charge Against Gold Fields (AP) ...
```

The harmful ones are business stories that AG News files under "World":
training on them teaches the model the wrong topic for this test article.

## Use it on your own data

You need three things:

- **Training examples** and **test examples**, each a TSV or CSV file with
  `question` and `answer` columns, where `answer` is a class label.
- **One or more LoRA adapters** fine-tuned on the training examples
  (`examples/quickstart.py` shows a minimal way to train them).

```bash
fast-trak run \
    --base_model Qwen/Qwen3.5-0.8B-Base \
    --checkpoints adapters/seed0 adapters/seed1 \
    --candidates train.tsv --targets test.tsv \
    --labels yes no maybe \
    --save_dir trak_store
```

This writes `trak_store/targets_scores.csv`, the training examples ranked from
most helpful to most harmful on average over the test examples. To keep the
best 5,000 as a new training set:

```bash
fast-trak select --scores trak_store/targets_scores.csv --n 5000 --out top5000.tsv
```

For large jobs, `fast-trak setup`, `featurize` and `score` run the same steps
separately, so each adapter can be processed as its own GPU job. The same
functions are available from Python; see [docs/design.md](docs/design.md).

## How it works

LoRA fine-tuning only trains small linear layers. For a linear layer, each
example's gradient can be rebuilt from two things PyTorch already computes
during a normal training step: the layer's input, and the gradient flowing
back into its output. FAST-TRAK records both with hooks and multiplies them
together per example. This is exact, not an approximation, and it needs no
special support from the model.

The rest of the speed comes from batching prompts by length, computing only
the output the score needs, and computing each gradient once even when it is
used several times. Details are in [docs/design.md](docs/design.md).

## How fast and how accurate?

Measured on one NVIDIA L40S with Qwen3.5-0.8B (rank-16 LoRA, prompts up to
512 tokens):

| | |
|---|---|
| Speed | 205 training examples per second |
| Compared with one example at a time | 5 per second, so 40x faster |
| Gradient error against one example at a time | 0.0006% (float32) |

Reproduce these with [`benchmarks/throughput.py`](benchmarks/throughput.py).

## Does picking data by score help?

FAST-TRAK was built for a thesis on legal text classification with little
real data. A large pool of synthetic training examples was scored, subsets
were chosen by score, and a small model was fine-tuned on each subset and
tested on real data from the Oral Argument Question Purpose task of
[LegalBench](https://hazyresearch.stanford.edu/legalbench/).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/oaqp_data_scaling_dark.png">
  <img src="docs/assets/oaqp_data_scaling_light.png" alt="Test accuracy against training-set size for the highest-scored, random and lowest-scored subsets. From 25k examples up, the highest-scored subsets reach 40 to 47 percent, random ones about 36 percent and the lowest-scored ones 21 to 28 percent.">
</picture>

Beyond 10,000 examples, the highest-scored subsets beat random subsets of the
same size by 3 to 10 points of accuracy, and the lowest-scored subsets are 8
to 17 points worse than random. Below 10,000, random is as good or better.

This figure comes from an earlier run on Qwen2.5-0.5B, scored with the
original TRAK code path, and will be replaced by a Qwen 3.5 run. The synthetic
pool is private and not part of this repository. On Qwen 3.5, FAST-TRAK's
scores on the same task match the thesis implementation to within the
difference between two identical runs.

## Limitations

- Only LoRA adapters are supported, not full fine-tuning.
- The task must be classification with a fixed set of labels, and no two
  labels may start with the same token.
- Scores are estimates. Before trusting a selected subset, compare it with a
  random subset of the same size.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The tests run on CPU in a few seconds.

## Credits

FAST-TRAK builds on [TRAK](https://github.com/MadryLab/trak) and uses its
projection and scoring code unchanged. If you use it, please cite the TRAK
paper:

```bibtex
@inproceedings{park2023trak,
  title     = {TRAK: Attributing Model Behavior at Scale},
  author    = {Park, Sung Min and Georgiev, Kristian and Ilyas, Andrew and Leclerc, Guillaume and Madry, Aleksander},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2023}
}
```

Released under the [MIT License](LICENSE).
