import distrax
from jax import numpy as jnp
from jax import random as jr

from surjectors._src.conditioners.mlp import make_mlp
from surjectors._src.surjectors.augment import Augment
from surjectors._src.testing.layer_harness import transform_layer


def _encoder_fn(n_dim):
  encoder_net = make_mlp([8, 8, n_dim * 2])

  def _fn(y):
    params = encoder_net(y)
    mu, log_scale = jnp.split(params, 2, -1)
    return distrax.Independent(distrax.Normal(mu, jnp.exp(log_scale)))

  return _fn


def _build(n_keep, n_extra):
  return Augment(n_keep, _encoder_fn(n_extra))


def test_inverse_augments_dimensions():
  n_keep, n_extra, n = 4, 2, 10
  y = jr.normal(jr.PRNGKey(0), (n, n_keep))

  init, apply = transform_layer(lambda: _build(n_keep, n_extra), "inverse")
  params = init(jr.PRNGKey(1), y)
  z, lc = apply(params, y, rng=jr.PRNGKey(2))

  assert z.shape == (n, n_keep + n_extra)
  assert jnp.array_equal(z[:, :n_keep], y)
  assert lc.shape == (n,)
  assert jnp.all(jnp.isfinite(lc))


def test_forward_drops_augmented_dimensions():
  n_keep, n_extra, n = 4, 2, 10
  z = jr.normal(jr.PRNGKey(0), (n, n_keep + n_extra))

  init, apply = transform_layer(lambda: _build(n_keep, n_extra), "forward")
  params = init(jr.PRNGKey(1), z)
  y, lc = apply(params, z)

  assert y.shape == (n, n_keep)
  assert jnp.array_equal(y, z[:, :n_keep])
  assert lc.shape == (n,)
  assert jnp.all(jnp.isfinite(lc))
