# Benchmarking MLX vs JAX/surjectors

Spike comparing a from-scratch MLX masked autoregressive flow against the
existing JAX/surjectors MAF for two-moons density-estimation training.

Design: `surjectors_docs/2026-07-06-benchmarking-mlx-spike-design.md`.

## Run

```bash
uv sync
uv run python main.py       # writes results/table.md and results/results.csv
uv run pytest --override-ini="addopts=" flow_mlx_test.py
```

(The `--override-ini` is needed because the surjectors root `pyproject.toml`
injects `--cov` addopts that break bare pytest in this sub-project.)

## What is measured

Three configurations — JAX/CPU, MLX/CPU, MLX/GPU (Metal). JAX-Metal is
omitted (experimental). Per configuration: steps/sec (throughput), total
train wall-clock, one-off compile/warmup time, and final test NLL. All runs
share architecture (2 affine MAF layers, `[32, 32]` MADE, reversing
permutation, standard-normal base), optimizer (Adam, lr 1e-4), batch size
(100), data, and step count (1000). Weight initialization differs between
frameworks by design; we compare speed and converged NLL, not identical
trajectories.

## Results

Apple Silicon, MLX 0.31.2, JAX 0.10.2. Representative run:

| framework | device | steps_per_sec | train_wall_s | warmup_s | init_nll | final_nll |
| --- | --- | --- | --- | --- | --- | --- |
| jax | cpu | 1.198e+04 | 0.08347 | 0.2028 | 2.847 | 2.621 |
| mlx | cpu | 3950 | 0.2532 | 0.04594 | 2.847 | 2.549 |
| mlx | gpu | 2433 | 0.4111 | 1.756 | 2.847 | 2.549 |

Both frameworks learn the density (final NLL < init NLL for all rows; the
`main()` NLL gate asserts this). Numbers vary run to run, especially
`warmup_s`, which depends on MLX's on-disk compilation cache — a cold cache
produced a ~12 s first-compile in isolation, versus sub-second once warm.

**Speed takeaway:** for this tiny 2D model, JAX/XLA on CPU dominates
(~3-5x faster than MLX). MLX on Metal is *slower* than MLX on CPU here: the
model is small enough that GPU dispatch overhead outweighs any parallelism
gain. This is the expected regime for a 2D flow with a 100-row batch — an MLX
GPU win, if it exists, would only appear at much larger model/batch sizes,
which this spike did not test.

## Ergonomics notes

Comparing writing `flow_mlx.py` from scratch against composing `flow_jax.py`
from surjectors.

**Autodiff.** MLX's `nn.value_and_grad(model, loss_fn)` differentiates with
respect to `model.trainable_parameters()` — model-aware and concise, no
explicit parameter pytree to thread. JAX/Haiku keeps params as an explicit
pytree returned by `init` and threaded through `apply`; `jax.value_and_grad`
operates on that pytree. Both are ergonomic; MLX hides the parameter container,
Haiku makes it explicit.

**Module system.** MLX `nn.Module` uses familiar mutable-object style and
auto-registers any `mx.array` attribute as a trainable parameter. That
convenience has a sharp edge: the MADE masks had to be stored as **NumPy**
arrays, not `mx.array`, otherwise MLX would register them as trainable and the
optimizer would corrupt the masking. Haiku's `hk.transform` purity avoids that
class of bug but costs the init/apply ceremony.

**Autoregressive masking.** This is the biggest gap. surjectors ships MADE and
`MaskedAutoregressive` (built on the TFP degree-mask helper), so the JAX flow
is pure composition. In MLX everything — degree assignment, mask construction,
`MaskedLinear`, the MADE forward, the affine inverse and log-det — was
hand-written. Correct, but it is real infrastructure that surjectors/distrax
provide for free.

**jit/compile friction.** JAX's `jax.jit` plus `block_until_ready` was
transparent with a small, consistent (~0.2 s) warmup. MLX's `mx.compile`
required explicitly capturing `[model.state, optimizer.state]` via
`inputs=`/`outputs=` and `mx.eval` at every timing boundary because execution
is lazy; the compile/warmup cost was also less predictable due to on-disk
caching.

**Verdict.** For porting surjectors, this spike gives no reason to switch:
MLX would require reimplementing the masking and bijector machinery that
surjectors and distrax already provide, and it showed no speed advantage on
this workload (and a GPU regression at this scale). MLX remains worth
re-testing only if the target workload moves to much larger models or batches
where Metal's unified-memory GPU could pay off.
