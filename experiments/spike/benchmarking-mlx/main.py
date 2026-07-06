"""Benchmark MLX vs JAX/surjectors MAF training across problem sizes.

Run: uv run python main.py [--size small|large]

`small` sits in the dispatch-bound regime (tiny 2D model); `large` is
compute-bound (wide, high-dimensional, big batch) where MLX's GPU can win.
Correctness is covered by flow_mlx_test.py and plot_samples.py, so this file
only measures speed.
"""

import argparse
import csv
import os
import statistics
import time
from dataclasses import dataclass
from functools import partial

import jax
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import optax
from jax import numpy as jnp, random as jr

from data import iter_batches, load_data
from flow_jax import make_flow as make_jax_flow
from flow_mlx import make_flow as make_mlx_flow


@dataclass(frozen=True)
class Config:
  """A benchmark problem size."""

  dim: int
  hidden_sizes: tuple[int, ...]
  n_layers: int
  batch_size: int
  n_data: int
  n_steps: int


SIZES = {
  "small": Config(2, (64, 64), 5, 100, 10_000, 100),
  "large": Config(64, (512, 512), 8, 8_192, 50_000, 100),
}
LR = 1e-4
N_WARMUP = 10
N_REPEATS = 1
SEED = 0


def _measure(train, cfg, fresh, step, sync):
  """Time a training loop: (steps/sec, median wall, first-compile time).

  Args:
    train: training array.
    cfg: the problem Config.
    fresh: ``fresh(seed)`` (re)builds the model and optimizer in place.
    step: ``step(batch)`` runs one optimizer step, returns the loss.
    sync: ``sync(loss)`` forces the (lazy/async) step to complete.
  """
  fresh(SEED)
  warm = iter_batches(train, cfg.batch_size, 1 + N_WARMUP, SEED)
  t0 = time.perf_counter()
  sync(step(next(warm)))
  warmup_s = time.perf_counter() - t0
  for batch in warm:
    loss = step(batch)
  sync(loss)

  walls = []
  for r in range(N_REPEATS):
    fresh(SEED + r)
    batches = iter_batches(train, cfg.batch_size, N_WARMUP + cfg.n_steps, r)
    for _ in range(N_WARMUP):
      loss = step(next(batches))
    sync(loss)
    t0 = time.perf_counter()
    for batch in batches:
      loss = step(batch)
    sync(loss)
    walls.append(time.perf_counter() - t0)

  wall = statistics.median(walls)
  return {
    "steps_per_sec": cfg.n_steps / wall,
    "train_wall_s": wall,
    "warmup_s": warmup_s,
  }


def benchmark_mlx(device, train, cfg):
  """Benchmark the MLX flow on the given device."""
  mx.set_default_device(device)
  ref = {}

  def fresh(seed):
    mx.random.seed(seed)
    model = make_mlx_flow(cfg.dim, cfg.hidden_sizes, cfg.n_layers)
    optimizer = optim.Adam(learning_rate=LR)
    mx.eval(model.parameters())
    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(y):
      loss, grads = nn.value_and_grad(model, _mlx_loss)(model, y)
      optimizer.update(model, grads)
      return loss

    ref["model"], ref["step"] = model, step

  metrics = _measure(
    train,
    cfg,
    fresh,
    lambda batch: ref["step"](mx.array(batch)),
    lambda loss: mx.eval(loss, ref["model"].state),
  )
  label = "gpu" if device == mx.gpu else "cpu"
  return {"framework": "mlx", "device": label, **metrics}


def _mlx_loss(model, y):
  return -mx.mean(model.log_prob(y))


def benchmark_jax(device, train, cfg):
  """Benchmark the surjectors flow on the given device ("cpu" or "gpu")."""
  jax_device = jax.devices("cpu" if device == "cpu" else "mps")[0]
  flow = make_jax_flow(cfg.dim, cfg.hidden_sizes, cfg.n_layers)
  optimizer = optax.adam(LR)
  ref = {}

  @jax.jit
  def step(params, opt_state, y):
    def loss_fn(p):
      return -jnp.mean(flow.apply(p, None, method="log_prob", y=y))

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return loss, optax.apply_updates(params, updates), opt_state

  def fresh(seed):
    params = flow.init(
      jr.key(seed), method="log_prob", y=jnp.asarray(train[: cfg.batch_size])
    )
    ref["params"], ref["opt"] = params, optimizer.init(params)

  def run(batch):
    loss, ref["params"], ref["opt"] = step(
      ref["params"], ref["opt"], jnp.asarray(batch)
    )
    return loss

  with jax.default_device(jax_device):
    metrics = _measure(
      train, cfg, fresh, run, lambda loss: loss.block_until_ready()
    )
  return {"framework": "jax", "device": device, **metrics}


def _write_results(rows, out_dir="results"):
  """Write rows to results.csv and table.md; return the markdown table."""
  os.makedirs(out_dir, exist_ok=True)
  cols = list(rows[0])
  with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=cols)
    writer.writeheader()
    writer.writerows(rows)

  fmt = lambda v: f"{v:.4g}" if isinstance(v, float) else str(v)
  lines = ["| " + " | ".join(cols) + " |", "|" + " --- |" * len(cols)]
  lines += ["| " + " | ".join(fmt(row[c]) for c in cols) + " |" for row in rows]
  table = "\n".join(lines) + "\n"
  with open(os.path.join(out_dir, "table.md"), "w") as f:
    f.write(table)
  return table


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--size", choices=SIZES, default="small")
  cfg = SIZES[parser.parse_args().size]
  train = load_data(cfg.dim, cfg.n_data, SEED)
  rows = [
    benchmark_jax("cpu", train, cfg),
    benchmark_jax("gpu", train, cfg),
    benchmark_mlx(mx.cpu, train, cfg),
    benchmark_mlx(mx.gpu, train, cfg),
  ]
  print(_write_results(rows))


if __name__ == "__main__":
  main()
