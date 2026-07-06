"""From-scratch MLX masked autoregressive flow (log_prob only)."""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

_LOG_2PI = math.log(2.0 * math.pi)


class DiagNormal:
  """Standard diagonal-Normal base distribution.

  Args:
    dim: event dimensionality.
  """

  def __init__(self, dim):
    self.dim = dim

  def log_prob(self, z):
    """Log density of a standard normal, summed over the last axis."""
    return mx.sum(-0.5 * (z**2 + _LOG_2PI), axis=-1)

  def sample(self, shape, key=None):
    """Draw standard-normal samples of shape ``(*shape, dim)``."""
    return mx.random.normal(shape=(*shape, self.dim), key=key)


def _made_masks(input_size, hidden_sizes, n_params):
  """Build degree-based MADE masks.

  Returns one mask per linear layer, each of shape (out_features,
  in_features), so a layer computes ``x @ (w * mask).T``.
  """
  degrees = [np.arange(1, input_size + 1)]
  span = max(input_size - 1, 1)
  for h in hidden_sizes:
    degrees.append(1 + (np.arange(h) % span))

  masks = []
  for d_prev, d_cur in zip(degrees[:-1], degrees[1:]):
    masks.append((d_cur[:, None] >= d_prev[None, :]).astype(np.float32))

  out_deg = degrees[0]
  d_last = degrees[-1]
  m_out = (out_deg[:, None] > d_last[None, :]).astype(np.float32)
  masks.append(np.repeat(m_out, n_params, axis=0))
  return masks


class MaskedLinear(nn.Module):
  """Linear layer whose weight is elementwise-masked by a constant.

  The mask is a NumPy array so MLX does not register it as a trainable
  parameter.

  Args:
    in_features: input dimensionality.
    out_features: output dimensionality.
    mask: NumPy array of shape (out_features, in_features).
  """

  def __init__(self, in_features, out_features, mask):
    super().__init__()
    self.weight = 0.01 * mx.random.normal(shape=(out_features, in_features))
    self.bias = mx.zeros((out_features,))
    self.mask = mask

  def __call__(self, x):
    return x @ (self.weight * self.mask).T + self.bias


class MADE(nn.Module):
  """Masked autoregressive density estimator conditioner.

  Maps ``(batch, input_size)`` to ``(batch, input_size, n_params)`` such
  that output row d depends only on inputs 0..d-1.

  Args:
    input_size: number of input features.
    hidden_sizes: sizes of the hidden layers.
    n_params: number of output parameters per input dimension.
  """

  def __init__(self, input_size, hidden_sizes, n_params):
    super().__init__()
    self.input_size = input_size
    self.n_params = n_params
    masks = _made_masks(input_size, hidden_sizes, n_params)
    sizes = [input_size, *hidden_sizes, input_size * n_params]
    self.layers = [
      MaskedLinear(sizes[i], sizes[i + 1], masks[i])
      for i in range(len(masks))
    ]

  def __call__(self, y):
    h = y
    for i, layer in enumerate(self.layers):
      h = layer(h)
      if i < len(self.layers) - 1:
        h = nn.relu(h)
    return h.reshape(y.shape[0], self.input_size, self.n_params)


class MAF(nn.Module):
  """Masked autoregressive flow over a diagonal-Normal base.

  Density-estimation direction only: ``log_prob`` maps data y to latent z
  via a single parallel MADE pass per layer, with a reversing permutation
  between layers.

  Args:
    dim: event dimensionality.
    hidden_sizes: MADE hidden-layer sizes.
    n_layers: number of MAF layers.
  """

  def __init__(self, dim, hidden_sizes, n_layers):
    super().__init__()
    self.dim = dim
    self.n_layers = n_layers
    self.base = DiagNormal(dim)
    self.mades = [
      MADE(dim, hidden_sizes, n_params=2) for _ in range(n_layers)
    ]

  def log_prob(self, y):
    """Log density of the data under the flow."""
    z = y
    ldj = mx.zeros((y.shape[0],))
    for i, made in enumerate(self.mades):
      params = made(z)
      mean = params[..., 0]
      log_scale = params[..., 1]
      z = (z - mean) * mx.exp(-log_scale)
      ldj = ldj - mx.sum(log_scale, axis=-1)
      if i < self.n_layers - 1:
        z = z[:, ::-1]
    return self.base.log_prob(z) + ldj
