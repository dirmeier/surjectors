"""A masked autoregressive flow in MLX.

Mirrors the surjectors design: every transform is a bijector exposing
``forward_and_log_det`` / ``inverse_and_log_det``, composed by ``Chain`` and
wrapped by ``TransformedDistribution``, so ``log_prob`` (density direction) and
``sample`` (generative direction) share the same primitives.
"""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

_LOG_2PI = math.log(2.0 * math.pi)


def _autoregressive_masks(input_size, hidden_sizes, n_params):
  degrees = [np.arange(1, input_size + 1)]
  span = max(input_size - 1, 1)
  for size in hidden_sizes:
    degrees.append(1 + (np.arange(size) % span))

  masks = [
    (nxt[:, None] >= prev[None, :]).astype(np.float32)
    for prev, nxt in zip(degrees[:-1], degrees[1:])
  ]
  out_mask = (degrees[0][:, None] > degrees[-1][None, :]).astype(np.float32)
  masks.append(np.repeat(out_mask, n_params, axis=0))
  return masks


class MaskedLinear(nn.Module):
  def __init__(self, in_features, out_features, mask):
    super().__init__()
    self.weight = 0.01 * mx.random.normal(shape=(out_features, in_features))
    self.bias = mx.zeros((out_features,))
    self.mask = mask

  def __call__(self, x):
    return x @ (self.weight * self.mask).T + self.bias


class MADE(nn.Module):
  def __init__(self, input_size, hidden_sizes, n_params):
    super().__init__()
    self.input_size = input_size
    self.n_params = n_params
    masks = _autoregressive_masks(input_size, hidden_sizes, n_params)
    sizes = [input_size, *hidden_sizes, input_size * n_params]
    self.layers = [
      MaskedLinear(sizes[i], sizes[i + 1], masks[i])
      for i in range(len(masks))
    ]

  def __call__(self, y):
    h = self.layers[0](y)
    for layer in self.layers[1:]:
      h = layer(nn.relu(h))
    return h.reshape(y.shape[0], self.input_size, self.n_params)


class Permutation:
  def __init__(self, order):
    self.order = order
    self.inverse_order = mx.argsort(order)

  def forward_and_log_det(self, z):
    return z[..., self.order], mx.zeros(z.shape[:-1])

  def inverse_and_log_det(self, y):
    return y[..., self.inverse_order], mx.zeros(y.shape[:-1])


class MaskedAutoregressive(nn.Module):
  def __init__(self, conditioner):
    super().__init__()
    self.conditioner = conditioner

  def inverse_and_log_det(self, y):
    loc, log_scale = self._params(y)
    return (y - loc) * mx.exp(-log_scale), -mx.sum(log_scale, axis=-1)

  def forward_and_log_det(self, z):
    y = mx.zeros_like(z)
    log_scale = mx.zeros_like(z)
    for _ in range(z.shape[-1]):
      loc, log_scale = self._params(y)
      y = z * mx.exp(log_scale) + loc
    return y, mx.sum(log_scale, axis=-1)

  def _params(self, x):
    params = self.conditioner(x)
    return params[..., 0], params[..., 1]


class Chain(nn.Module):
  def __init__(self, bijectors):
    super().__init__()
    self.bijectors = bijectors

  def inverse_and_log_det(self, y):
    log_det = mx.zeros(y.shape[:-1])
    for bijector in self.bijectors:
      y, contribution = bijector.inverse_and_log_det(y)
      log_det = log_det + contribution
    return y, log_det

  def forward_and_log_det(self, z):
    log_det = mx.zeros(z.shape[:-1])
    for bijector in reversed(self.bijectors):
      z, contribution = bijector.forward_and_log_det(z)
      log_det = log_det + contribution
    return z, log_det


class DiagNormal:
  def __init__(self, dim):
    self.dim = dim

  def log_prob(self, z):
    return mx.sum(-0.5 * (z**2 + _LOG_2PI), axis=-1)

  def sample(self, n, key=None):
    return mx.random.normal(shape=(n, self.dim), key=key)


class TransformedDistribution(nn.Module):
  def __init__(self, base, transform):
    super().__init__()
    self.base = base
    self.transform = transform

  def log_prob(self, y):
    z, log_det = self.transform.inverse_and_log_det(y)
    return self.base.log_prob(z) + log_det

  def sample(self, n, key=None):
    y, _ = self.transform.forward_and_log_det(self.base.sample(n, key=key))
    return y


def make_flow(dim, hidden_sizes, n_layers):
  order = mx.arange(dim - 1, -1, -1)
  bijectors = []
  for i in range(n_layers):
    bijectors.append(MaskedAutoregressive(MADE(dim, hidden_sizes, 2)))
    if i < n_layers - 1:
      bijectors.append(Permutation(order))
  return TransformedDistribution(DiagNormal(dim), Chain(bijectors))
