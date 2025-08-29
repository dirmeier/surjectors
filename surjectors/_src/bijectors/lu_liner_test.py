# pylint: skip-file

import haiku as hk
from jax import numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from surjectors import LULinear, TransformedDistribution


@hk.without_apply_rng
@hk.transform
def fn(**kwargs):
  base_distribution = tfd.Independent(
    tfd.Normal(jnp.zeros(5), jnp.ones(5)),
    reinterpreted_batch_ndims=1,
  )
  td = TransformedDistribution(base_distribution, LULinear(5))
  return td.log_prob(**kwargs)


def test_lu_linear():
  y = jr.normal(jr.PRNGKey(1), shape=(10, 5))
  params = fn.init(jr.key(0), y=y)
  _ = fn.apply(params, y=y)
