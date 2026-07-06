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
