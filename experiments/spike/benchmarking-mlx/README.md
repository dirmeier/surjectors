# MLX vs JAX for normalizing flows

A benchmark evaluating if MLX is worth using over JAX as an array framework for `surjectors`.
`flow_mlx.py` is a small MAF written in MLX. `flow_jax.py` builds the same flow with surjectors.

## Running it

```bash
uv sync
uv run python main.py --size small   # or large
```

## Numbers

M1, MLX 0.31.2, JAX 0.10.2 (jax-mps), Python 3.13.

**small** — 2D two-moons, `(64,64)` MADE, 5 layers, batch 100:

| framework | device | steps/sec | train_wall_s | warmup_s |
| --- | --- | --- | --- | --- |
| jax | cpu | 3117 | 0.03208 | 0.3019 |
| jax | gpu | 511.0 | 0.1957 | 2.289 |
| mlx | cpu | 1350 | 0.07405 | 0.04999 |
| mlx | gpu | 961.8 | 0.1040 | 0.02418 |

**large** — 64D, `(512,512)` MADE, 8 layers, batch 8192:

| framework | device | steps/sec | train_wall_s | warmup_s |
| --- | --- | --- | --- | --- |
| jax | cpu | 2.901 | 34.48 | 0.9856 |
| jax | gpu | 15.81 | 6.325 | 0.7462 |
| mlx | cpu | 2.598 | 38.49 | 0.5830 |
| mlx | gpu | 15.48 | 6.459 | 0.1033 |

## Summary

- Small model: JAX on CPU wins, both GPUs slower than CPUs.
- Large model: JAX/GPU ≈ MLX/GPU, GPU wins.
- jax-mps gives JAX the GPU with no downgrade, so MLX has no speed edge.
- No performance reason to move surjectors to MLX.
