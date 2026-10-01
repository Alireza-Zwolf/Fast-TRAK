# The `fast_trak` package

What each module does.

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

## Where to start reading

1. `gradients.py`: the core idea, per-example gradients from one backward pass.
2. `outputs.py`: what is differentiated, the model's margin on the correct label.
3. `traker.py`: how each gradient is projected several times and stored.
4. `pipeline.py`: how the pieces are put together into the three stages.

The method itself is explained in [docs/design.md](../../docs/design.md).
