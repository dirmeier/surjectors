import distrax
from jax import numpy as jnp
from jax import random as jr

from surjectors._src.conditioners.mlp import make_mlp
from surjectors._src.surjectors.masked_coupling_inference_funnel import (
  MaskedCouplingInferenceFunnel,
)
from surjectors._src.testing.layer_harness import transform_layer


def _decoder_fn(n_dim):
  decoder_net = make_mlp([8, 8, n_dim * 2])

  def _fn(z):
    params = decoder_net(z)
    mu, log_scale = jnp.split(params, 2, -1)
    return distrax.Independent(distrax.Normal(mu, jnp.exp(log_scale)))

  return _fn


def _bijector_fn(params):
  shift, log_scale = jnp.split(params, 2, -1)
  return distrax.ScalarAffine(shift, jnp.exp(log_scale))


def _build(n_dimension, n_keep):
  return MaskedCouplingInferenceFunnel(
    n_keep,
    _decoder_fn(n_dimension - n_keep),
    make_mlp([8, 8, n_dimension * 2]),
    _bijector_fn,
  )


def test_inverse_reduces_dimensionality():
  n_dimension, n_keep, n = 6, 4, 10
  y = jr.normal(jr.PRNGKey(0), (n, n_dimension))

  init, apply = transform_layer(
    lambda: _build(n_dimension, n_keep), "inverse"
  )
  params = init(jr.PRNGKey(1), y)
  z, lc = apply(params, y)

  assert z.shape == (n, n_keep)
  assert lc.shape == (n,)
  assert jnp.all(jnp.isfinite(lc))


def test_forward_reconstructs_full_dimensionality():
  n_dimension, n_keep, n = 6, 4, 10
  z = jr.normal(jr.PRNGKey(0), (n, n_keep))

  init, apply = transform_layer(lambda: _build(n_dimension, n_keep), "forward")
  params = init(jr.PRNGKey(1), z)
  y, lc = apply(params, z, rng=jr.PRNGKey(2))

  assert y.shape == (n, n_dimension)
  assert lc.shape == (n,)
  assert jnp.all(jnp.isfinite(lc))


def test_declares_forward_needs_rng():
  assert MaskedCouplingInferenceFunnel.forward_needs_rng is True
  assert MaskedCouplingInferenceFunnel.inverse_needs_rng is False
