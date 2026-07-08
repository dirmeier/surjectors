from collections.abc import Callable
from typing import ClassVar

import haiku as hk
from jax import numpy as jnp

from surjectors._src.surjectors.funnel import Funnel


class Slice(Funnel):
  """A slice funnel.

  The forward direction invents the dropped dimensions by sampling from
  the decoder, so it needs an active Haiku rng
  (``forward_needs_rng`` is ``True``); the inverse direction is a pure
  likelihood evaluation and needs no rng.

  Args:
      n_keep: number of dimensions to keep
      decoder: callable

  Examples:
      >>> import haiku as hk
      >>> from jax import random as jr
      >>> from tensorflow_probability.substrates.jax import distributions as tfd
      >>> from surjectors.nn import make_mlp
      >>> from surjectors import Slice, TransformedDistribution

      >>> def decoder_fn(n_dim):
      ...   def fn(z):
      ...     params = make_mlp((64, 64, n_dim * 2))(z)
      ...     mu, log_scale = jnp.split(params, 2, -1)
      ...     return tfd.Independent(
      ...       tfd.Normal(mu, jnp.exp(log_scale))
      ...     )
      ...   return fn
      >>>
      >>> @hk.without_apply_rng
      ... @hk.transform
      ... def fn(inputs):
      ...   base_distribution = tfd.Independent(
      ...     tfd.Normal(jnp.zeros(4), jnp.ones(4)),
      ...     reinterpreted_batch_ndims=1,
      ...   )
      ...   td = TransformedDistribution(
      ...     base_distribution,
      ...     Slice(4, decoder_fn(10 - 4))
      ...   )
      ...   return td.log_prob(inputs)
      >>> data = jr.normal(jr.PRNGKey(1), shape=(10, 10))
      >>> params = fn.init(jr.key(0), data)
      >>> lps = fn.apply(params, data)

  References:
    .. [1] Nielsen, Didrik, et al. "SurVAE Flows: Surjections to Bridge the
       Gap between VAEs and Flows". Advances in Neural Information
       Processing Systems, 2020.
  """

  forward_needs_rng: ClassVar[bool] = (
    True  # invents the dropped dims by sampling
  )

  def __init__(self, n_keep: int, decoder: Callable):
    super().__init__(n_keep)
    self.decoder = decoder

  def _inverse_and_likelihood_contribution(self, y, x=None, **kwargs):
    z, y_minus = self._split_input(y)
    z_condition = self._condition(z, x)
    lc = self.decoder(z_condition).log_prob(y_minus)
    return z, lc

  def _forward_and_likelihood_contribution(self, z, x=None, **kwargs):
    z_condition = self._condition(z, x)
    y_minus, lc = self.decoder(z_condition).sample_and_log_prob(
      seed=hk.next_rng_key()
    )
    y = jnp.concatenate([z, y_minus], axis=-1)
    return y, lc
