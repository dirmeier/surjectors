import haiku as hk
from jax import numpy as jnp

from surjectors._src.surjectors.funnel import Funnel


class Augment(Funnel):
  """Augment generative funnel.

  Args:
      n_keep: number of dimensions to keep
      encoder: encoder callable
  """

  def __init__(self, n_keep, encoder):
    super().__init__(n_keep)
    self.encoder = encoder

  def _inverse_and_likelihood_contribution(self, y, x=None, **kwargs):
    z_plus = y
    y_condition = self._condition(y, x)
    z_minus, lc = self.encoder(y_condition).sample_and_log_prob(
      seed=hk.next_rng_key()
    )
    z = jnp.concatenate([z_plus, z_minus], axis=-1)
    return z, -lc

  def _forward_and_likelihood_contribution(self, z, x=None, **kwargs):
    z_plus, z_minus = self._split_input(z)
    y = z_plus
    y_condition = self._condition(z_plus, x)
    lc = self.encoder(y_condition).log_prob(z_minus)
    return y, -lc
