import numpy as np
from sklearn.datasets import make_moons


def load_two_moons(n_data, noise, seed):
  x, _ = make_moons(n_samples=n_data, noise=noise, random_state=seed)
  x = x.astype(np.float32)
  n_test = n_data // 10
  test, train = x[:n_test], x[n_test:]
  mean = train.mean(axis=0, keepdims=True)
  std = train.std(axis=0, keepdims=True)
  train = (train - mean) / std
  test = (test - mean) / std
  return train, test


def load_data(dim, n_data, seed, noise=0.1):
  if dim == 2:
    return load_two_moons(n_data, noise, seed)[0]
  rng = np.random.default_rng(seed)
  return rng.standard_normal((n_data, dim)).astype(np.float32)


def iter_batches(data, batch_size, n_steps, seed):
  rng = np.random.default_rng(seed)
  n = data.shape[0]
  for _ in range(n_steps):
    idx = rng.integers(0, n, size=batch_size)
    yield data[idx]
