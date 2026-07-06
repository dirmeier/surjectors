"""Unconditional MAF built from the existing surjectors library."""

import haiku as hk
from jax import numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd

from surjectors import (
  Chain,
  MaskedAutoregressive,
  Permutation,
  ScalarAffine,
  TransformedDistribution,
)
from surjectors.nn import MADE
from surjectors.util import unstack


def make_flow(dim, hidden_sizes, n_layers):
  """Return an hk.transform of an unconditional two-moons MAF.

  The architecture mirrors the MLX flow: n_layers affine MAF layers with a
  reversing permutation between them and a standard-normal base.

  Args:
    dim: event dimensionality.
    hidden_sizes: MADE hidden-layer sizes.
    n_layers: number of MAF layers.

  Returns:
    an hk.transform whose apply takes (params, rng, method, **kwargs).
  """

  def _bijector_fn(params):
    means, log_scales = unstack(params, -1)
    return ScalarAffine(means, jnp.exp(log_scales))

  def _flow(method, **kwargs):
    layers = []
    order = jnp.arange(dim)
    for _ in range(n_layers):
      layers.append(
        MaskedAutoregressive(
          bijector_fn=_bijector_fn,
          conditioner=MADE(
            dim,
            list(hidden_sizes),
            2,
            w_init=hk.initializers.TruncatedNormal(0.01),
            b_init=jnp.zeros,
          ),
        )
      )
      order = order[::-1]
      layers.append(Permutation(order, 1))
    layers = layers[:-1]
    base = tfd.Independent(
      tfd.Normal(jnp.zeros(dim), jnp.ones(dim)),
      reinterpreted_batch_ndims=1,
    )
    td = TransformedDistribution(base, Chain(layers))
    return td(method=method, **kwargs)

  return hk.transform(_flow)
