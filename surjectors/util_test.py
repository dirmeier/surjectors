import distrax
import haiku as hk
import jax
import optax
from jax import numpy as jnp
from jax import random

from surjectors._src.distributions.transformed_distribution import (
  TransformedDistribution,
)
from surjectors._src.surjectors.slice import Slice
from surjectors.util import as_model


def _decoder_fn(z):
  return distrax.Independent(
    distrax.Normal(jnp.zeros_like(z), jnp.ones_like(z)),
    reinterpreted_batch_ndims=1,
  )


def _build_td():
  base = distrax.Independent(
    distrax.Normal(jnp.zeros(2), jnp.ones(2)), reinterpreted_batch_ndims=1
  )
  return TransformedDistribution(base, Slice(2, _decoder_fn))


def test_as_model_trains_and_samples():
  model = as_model(_build_td, methods=("log_prob", "sample"))
  rng_seq = hk.PRNGSequence(0)
  y = random.normal(next(rng_seq), (8, 4))

  params = model.init(next(rng_seq), y=y)

  def loss_fn(params, rng, y):
    lp = model.apply.log_prob(params, rng, y=y)
    return -jnp.sum(lp)

  loss, grads = jax.value_and_grad(loss_fn)(params, next(rng_seq), y)
  adam = optax.adamw(0.001)
  state = adam.init(params)
  updates, _ = adam.update(grads, state, params)
  new_params = optax.apply_updates(params, updates)

  sample = model.apply.sample(new_params, next(rng_seq), sample_shape=(8,))
  assert jnp.isfinite(loss)
  assert sample.shape == (8, 4)
