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
  def _bijector_fn(params):
    means, log_scales = unstack(params, -1)
    return ScalarAffine(means, jnp.exp(log_scales))

  def _flow(method, **kwargs):
    order = jnp.arange(dim)[::-1]
    layers = []
    for i in range(n_layers):
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
      if i < n_layers - 1:
        layers.append(Permutation(order, 1))
    base = tfd.Independent(
      tfd.Normal(jnp.zeros(dim), jnp.ones(dim)),
      reinterpreted_batch_ndims=1,
    )
    td = TransformedDistribution(base, Chain(layers))
    return td(method=method, **kwargs)

  return hk.transform(_flow)
