# Design notes

How FAST-TRAK computes TRAK scores, and where each piece lives.

## The estimator

For a model output function `f(z; θ)` and a training set of `N` examples,
TRAK scores training example `i` against target `j` as

```text
φ(z)   = P^T ∇θ f(z; θ) / sqrt(D)              projected gradient, D = number of weights
Φ      = [φ(z_1); ...; φ(z_N)]                  N x k feature matrix
τ(i,j) = [ Φ (Φ^T Φ)^-1 φ(z_j) ]_i · Q_i        Q_i = output-to-loss factor of example i
```

and averages over an ensemble of trained checkpoints. `P` is a random
`D x k` projection. FAST-TRAK computes `∇θ f` and `Q`; everything after that
is TRAK's own code.

## Model output: the answer margin

A causal LM is used as a classifier by reading the first answer token. With
`z_c` the logit of the first token of label `c` at the position where the
answer begins, and `y` the correct label:

```text
f = z_y - logsumexp_{c != y} z_c
Q = 1 - softmax(z)_y            (softmax over the label tokens only)
```

This is the multi-class margin of the TRAK paper restricted to the label
vocabulary. `Q` is evaluated in float32 regardless of the model's dtype.
Implemented in `outputs.py`; label tokens are resolved in `labels.py`, in
prompt context, and must be pairwise distinct.

## Per-example gradients without vmap

`gradients.py` holds the core of the package. For every trainable LoRA
`A`/`B` linear layer:

1. a forward hook stores the layer input `x`, shaped `[batch, tokens, in]`;
2. a tensor hook on the layer output receives `∂(Σ_b f_b)/∂y`, shaped
   `[batch, tokens, out]`, during the single backward pass;
3. `einsum("bto,bti->boi", dy, x)` gives the `[batch, out, in]` per-example
   gradient of that layer's weight.

Row `b` of `∂(Σ_b f_b)/∂y` equals `∂f_b/∂y_b` because example `b`'s output
depends only on its own row. That holds whenever examples do not interact
inside the model, which is true of padded batches with an attention mask. It
would not hold for sequence packing, where several examples share a row.

A layer called more than once per forward pass contributes the sum of its
calls. Any trainable parameter that is not a LoRA linear weight raises
immediately.

## Batching

`data.py` tokenises each prompt once, left-truncates to `max_length` so the
answer position is preserved, and left-pads within a batch so that position is
always the last one. `TokenBudgetBatchSampler` sorts by length and fills each
batch up to a padded-token budget. Results do not depend on batch
composition, which is also why an out-of-memory batch can simply be halved
(`run_with_oom_split` in `traker.py`).

For Qwen 3.5, batch widths are rounded up to a multiple of 64, the chunk size
of its DeltaNet layers, and the token budget defaults to a value measured per
model size (`qwen35.py`).

## Several projections per checkpoint

Averaging over independent projections reduces projection noise. In stock
TRAK, each projection is a separate model ID and a separate pass over the
data. `MultiProjectionTRAKer` (`traker.py`) computes each batch gradient once
and projects it `num_projections` times.

Each `(checkpoint, projection)` pair is stored under the virtual model ID
`checkpoint * num_projections + projection`, so TRAK's saver, feature
finalisation and score aggregation need no changes. The resulting store is
byte-identical to what stock TRAK writes when run once per virtual ID, which
the test suite asserts.

## Stages and the store

| Stage | Cost | Parallelism |
|---|---|---|
| `setup` | seconds | run once, first |
| `featurize` | one pass over the candidates per checkpoint | one job per checkpoint |
| `score` | one pass over the targets per checkpoint, then linear algebra | repeat per target set |

The store is TRAK's memory-mapped layout plus a `metadata.json` that records
the base model, dtype, labels, prompt template, truncation length, padding
multiple, ensemble layout and a hash of the candidate file. Opening a store
with any of these changed is an error (`store.py`), since features computed
under one setting are meaningless under another.

## Using the pieces directly

The gradient computer plugs into stock TRAK for any model output that can be
written as one scalar per example:

```python
from trak import TRAKer
from fast_trak import BatchedLoRAGradientComputer
from fast_trak.projection import build_projector

class MyOutput:
    def forward_batched(self, model, batch):
        ...                               # one differentiable scalar per example
        return outputs, loss_factor       # shapes [batch] and [batch, 1]

traker = TRAKer(
    model=peft_model,
    task=MyOutput(),
    train_set_size=n,
    gradient_computer=BatchedLoRAGradientComputer,
    grad_wrt=[name for name, p in peft_model.named_parameters() if p.requires_grad],
    projector=build_projector(grad_dim, 2048, seed=0, device="cuda"),
)
```

## Module map

| Module | Responsibility |
|---|---|
| `gradients.py` | hook-based exact per-example LoRA gradients |
| `outputs.py` | answer-margin model output and its loss factor |
| `traker.py` | multi-projection TRAKer, OOM-split helper |
| `projection.py` | projector sized to the adapter gradient |
| `data.py` | file reading, tokenisation, token-budget batching |
| `labels.py` | label to first-token resolution |
| `models.py` | base model and adapter loading |
| `qwen35.py` | Qwen 3.5 kernel selection and batching defaults |
| `store.py` | store metadata validation |
| `pipeline.py` | the `setup`, `featurize` and `score` stages |
| `selection.py` | top, bottom and random subsets from a ranking |
| `cli.py` | the `fast-trak` command |
