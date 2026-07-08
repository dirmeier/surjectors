import distrax
from jax import numpy as jnp
from jax import random as jr

from surjectors._src.conditioners.mlp import make_mlp
from surjectors._src.surjectors.slice import Slice
from surjectors._src.testing.layer_harness import transform_layer


def _decoder_fn(n_dim):
  decoder_net = make_mlp([8, 8, n_dim * 2])

  def _fn(z):
    params = decoder_net(z)
    mu, log_scale = jnp.split(params, 2, -1)
    return distrax.Independent(distrax.Normal(mu, jnp.exp(log_scale)))

  return _fn


def _build(n_dimension, n_keep):
  return Slice(n_keep, _decoder_fn(n_dimension - n_keep))


def test_inverse_drops_dimensions():
  n_dimension, n_keep, n = 6, 4, 10
  y = jr.normal(jr.PRNGKey(0), (n, n_dimension))

  init, apply = transform_layer(lambda: _build(n_dimension, n_keep), "inverse")
  params = init(jr.PRNGKey(1), y)
  z, lc = apply(params, y)

  assert z.shape == (n, n_keep)
  assert jnp.array_equal(z, y[:, :n_keep])
  assert lc.shape == (n,)
  assert jnp.all(jnp.isfinite(lc))


def test_forward_reconstructs_kept_dimensions():
  n_dimension, n_keep, n = 6, 4, 10
  z = jr.normal(jr.PRNGKey(0), (n, n_keep))

  init, apply = transform_layer(lambda: _build(n_dimension, n_keep), "forward")
  params = init(jr.PRNGKey(1), z)
  y, lc = apply(params, z, rng=jr.PRNGKey(3))

  assert y.shape == (n, n_dimension)
  assert jnp.array_equal(y[:, :n_keep], z)
  assert lc.shape == (n,)
  assert jnp.all(jnp.isfinite(lc))
