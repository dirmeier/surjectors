import distrax
from jax import numpy as jnp
from jax import random as jr

from surjectors._src.bijectors.masked_coupling import MaskedCoupling
from surjectors._src.conditioners.mlp import make_mlp
from surjectors._src.testing.layer_harness import transform_layer
from surjectors.util import make_alternating_binary_mask


def _bijector_fn(params):
  shift, log_scale = jnp.split(params, 2, -1)
  return distrax.ScalarAffine(shift, jnp.exp(log_scale))


def _build(n_dimension):
  mask = make_alternating_binary_mask(n_dimension, True)
  return MaskedCoupling(mask, make_mlp([8, 8, n_dimension * 2]), _bijector_fn)


def test_round_trip():
  n_dimension, n = 4, 10
  y = jr.normal(jr.PRNGKey(0), (n, n_dimension))

  inv_init, inv_apply = transform_layer(lambda: _build(n_dimension), "inverse")
  params = inv_init(jr.PRNGKey(1), y)
  z, ldj_inv = inv_apply(params, y)

  _, fwd_apply = transform_layer(lambda: _build(n_dimension), "forward")
  y_hat, ldj_fwd = fwd_apply(params, z)

  assert z.shape == (n, n_dimension)
  assert ldj_inv.shape == (n,)
  assert jnp.allclose(y_hat, y, atol=1e-4)
  assert jnp.allclose(ldj_inv, -ldj_fwd, atol=1e-4)


def test_conditional_round_trip():
  n_dimension, n_condition, n = 4, 2, 10
  y = jr.normal(jr.PRNGKey(0), (n, n_dimension))
  x = jr.normal(jr.PRNGKey(3), (n, n_condition))

  inv_init, inv_apply = transform_layer(lambda: _build(n_dimension), "inverse")
  params = inv_init(jr.PRNGKey(1), y, x=x)
  z, ldj_inv = inv_apply(params, y, x=x)

  _, fwd_apply = transform_layer(lambda: _build(n_dimension), "forward")
  y_hat, ldj_fwd = fwd_apply(params, z, x=x)

  assert jnp.allclose(y_hat, y, atol=1e-4)
  assert jnp.allclose(ldj_inv, -ldj_fwd, atol=1e-4)
