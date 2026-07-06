"""Two-moons dataset: standardized, split, and batched."""

import numpy as np
from sklearn.datasets import make_moons


def load_two_moons(n_data, noise, seed):
  """Return standardized (train, test) two-moons arrays as float32.

  Args:
    n_data: total number of points (split 90/10 train/test).
    noise: gaussian noise passed to make_moons.
    seed: random seed for reproducibility.

  Returns:
    (train, test) float32 arrays of shape (n_train, 2) and (n_test, 2),
    standardized using the training-set mean and std.
  """
  x, _ = make_moons(n_samples=n_data, noise=noise, random_state=seed)
  x = x.astype(np.float32)
  n_test = n_data // 10
  test, train = x[:n_test], x[n_test:]
  mean = train.mean(axis=0, keepdims=True)
  std = train.std(axis=0, keepdims=True)
  train = (train - mean) / std
  test = (test - mean) / std
  return train, test


def iter_batches(data, batch_size, n_steps, seed):
  """Yield n_steps random minibatches (with replacement across steps).

  Args:
    data: float32 array of shape (n, 2).
    batch_size: size of each yielded batch.
    n_steps: number of batches to yield.
    seed: seed for the batch-index RNG.

  Yields:
    float32 arrays of shape (batch_size, 2).
  """
  rng = np.random.default_rng(seed)
  n = data.shape[0]
  for _ in range(n_steps):
    idx = rng.integers(0, n, size=batch_size)
    yield data[idx]
