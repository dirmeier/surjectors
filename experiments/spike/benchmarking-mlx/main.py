"""Benchmark MLX vs JAX/surjectors MAF training on two-moons.

Run: uv run python main.py
"""

import csv
import os
import statistics
import time
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

import jax
import optax
from jax import numpy as jnp, random as jr

from data import iter_batches, load_two_moons
from flow_jax import make_flow
from flow_mlx import MAF

DIM = 2
HIDDEN_SIZES = (32, 32)
N_LAYERS = 2
LR = 1e-4
BATCH_SIZE = 100
N_DATA = 10_000
N_STEPS = 1_000
N_WARMUP = 20
N_REPEATS = 3
NOISE = 0.1
SEED = 0


def _mlx_nll(model, data):
  return float(-mx.mean(model.log_prob(mx.array(data))))


def benchmark_mlx(device, train, test):
  """Train the MLX MAF once and return a metrics dict.

  Args:
    device: mx.cpu or mx.gpu.
    train: standardized training array.
    test: standardized test array.

  Returns:
    dict with framework, device, steps_per_sec, train_wall_s, warmup_s,
    init_nll, final_nll.
  """
  mx.set_default_device(device)

  def build():
    mx.random.seed(SEED)
    model = MAF(DIM, HIDDEN_SIZES, N_LAYERS)
    mx.eval(model.parameters())
    optimizer = optim.Adam(learning_rate=LR)

    def loss_fn(model, y):
      return -mx.mean(model.log_prob(y))

    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(y):
      loss, grads = nn.value_and_grad(model, loss_fn)(model, y)
      optimizer.update(model, grads)
      return loss

    return model, step

  # compile/warmup: first call includes tracing + compilation.
  model, step = build()
  init_nll = _mlx_nll(model, test)
  batches = iter_batches(train, BATCH_SIZE, 1 + N_WARMUP + N_STEPS, SEED)
  t0 = time.perf_counter()
  loss = step(mx.array(next(batches)))
  mx.eval(loss, model.state, model.parameters())
  warmup_s = time.perf_counter() - t0
  for _ in range(N_WARMUP):
    loss = step(mx.array(next(batches)))
  mx.eval(loss, model.state)

  # timed repeats: rebuild fresh each repeat, measure N_STEPS.
  wall_times = []
  final_nll = init_nll
  for r in range(N_REPEATS):
    model, step = build()
    rb = iter_batches(train, BATCH_SIZE, N_WARMUP + N_STEPS, SEED + r)
    for _ in range(N_WARMUP):
      step(mx.array(next(rb)))
    mx.eval(model.state)
    t0 = time.perf_counter()
    for _ in range(N_STEPS):
      loss = step(mx.array(next(rb)))
    mx.eval(loss, model.state, model.parameters())
    wall_times.append(time.perf_counter() - t0)
    final_nll = _mlx_nll(model, test)

  train_wall_s = statistics.median(wall_times)
  return {
    "framework": "mlx",
    "device": "gpu" if device == mx.gpu else "cpu",
    "steps_per_sec": N_STEPS / train_wall_s,
    "train_wall_s": train_wall_s,
    "warmup_s": warmup_s,
    "init_nll": init_nll,
    "final_nll": final_nll,
  }


def _jax_nll(flow, params, data):
  lp = flow.apply(params, None, method="log_prob", y=jnp.asarray(data))
  return float(-jnp.mean(lp))


def benchmark_jax(train, test):
  """Train the surjectors MAF on CPU and return a metrics dict.

  Args:
    train: standardized training array.
    test: standardized test array.

  Returns:
    dict with the same keys as benchmark_mlx (device is always "cpu").
  """
  jax.config.update("jax_platform_name", "cpu")
  flow = make_flow(DIM, HIDDEN_SIZES, N_LAYERS)

  def build():
    params = flow.init(
      jr.key(SEED), method="log_prob", y=jnp.asarray(train[:BATCH_SIZE])
    )
    optimizer = optax.adam(LR)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, y):
      def loss_fn(p):
        return -jnp.mean(flow.apply(p, None, method="log_prob", y=y))

      loss, grads = jax.value_and_grad(loss_fn)(params)
      updates, opt_state = optimizer.update(grads, opt_state, params)
      return loss, optax.apply_updates(params, updates), opt_state

    return params, opt_state, step

  params, opt_state, step = build()
  init_nll = _jax_nll(flow, params, test)
  batches = iter_batches(train, BATCH_SIZE, 1 + N_WARMUP + N_STEPS, SEED)

  t0 = time.perf_counter()
  loss, params, opt_state = step(params, opt_state, jnp.asarray(next(batches)))
  loss.block_until_ready()
  warmup_s = time.perf_counter() - t0
  for _ in range(N_WARMUP):
    loss, params, opt_state = step(
      params, opt_state, jnp.asarray(next(batches))
    )
  loss.block_until_ready()

  wall_times = []
  final_nll = init_nll
  for r in range(N_REPEATS):
    params, opt_state, step = build()
    rb = iter_batches(train, BATCH_SIZE, N_WARMUP + N_STEPS, SEED + r)
    for _ in range(N_WARMUP):
      loss, params, opt_state = step(
        params, opt_state, jnp.asarray(next(rb))
      )
    loss.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(N_STEPS):
      loss, params, opt_state = step(
        params, opt_state, jnp.asarray(next(rb))
      )
    loss.block_until_ready()
    wall_times.append(time.perf_counter() - t0)
    final_nll = _jax_nll(flow, params, test)

  train_wall_s = statistics.median(wall_times)
  return {
    "framework": "jax",
    "device": "cpu",
    "steps_per_sec": N_STEPS / train_wall_s,
    "train_wall_s": train_wall_s,
    "warmup_s": warmup_s,
    "init_nll": init_nll,
    "final_nll": final_nll,
  }


_COLUMNS = [
  "framework",
  "device",
  "steps_per_sec",
  "train_wall_s",
  "warmup_s",
  "init_nll",
  "final_nll",
]


def _write_results(rows, out_dir="results"):
  os.makedirs(out_dir, exist_ok=True)
  with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)

  lines = ["| " + " | ".join(_COLUMNS) + " |"]
  lines.append("| " + " | ".join("---" for _ in _COLUMNS) + " |")
  for row in rows:
    cells = []
    for col in _COLUMNS:
      v = row[col]
      cells.append(f"{v:.4g}" if isinstance(v, float) else str(v))
    lines.append("| " + " | ".join(cells) + " |")
  table = "\n".join(lines) + "\n"
  with open(os.path.join(out_dir, "table.md"), "w") as f:
    f.write(table)
  return table


def main():
  train, test = load_two_moons(N_DATA, NOISE, SEED)
  rows = [
    benchmark_jax(train, test),
    benchmark_mlx(mx.cpu, train, test),
    benchmark_mlx(mx.gpu, train, test),
  ]
  for row in rows:
    assert np.isfinite(row["final_nll"]), row
    assert row["final_nll"] < row["init_nll"], row
  table = _write_results(rows)
  print(table)


if __name__ == "__main__":
  main()
