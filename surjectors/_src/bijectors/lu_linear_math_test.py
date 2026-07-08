import jax
import pytest
from jax import numpy as jnp
from jax import random as jr

from surjectors._src.bijectors.lu_linear import LULinear
from surjectors._src.testing.layer_harness import transform_layer


def _build(n_dimension):
  return LULinear(n_dimension)


def test_inverse_is_linear_and_log_det_matches_diagonal():
  n_dimension, n = 4, 10
  y = jr.normal(jr.PRNGKey(0), (n, n_dimension))

  init, apply = transform_layer(lambda: _build(n_dimension), "inverse")
  params = init(jr.PRNGKey(1), y)
  z, lc = apply(params, y)

  assert z.shape == (n, n_dimension)
  assert lc.shape == (n,)

  def _log_det(params):
    diag_entries = params["lu_linear"]["diag_entries"]
    upper_diag = jax.nn.softplus(diag_entries) + 1e-4
    return jnp.sum(jnp.log(upper_diag))

  expected = _log_det(params)
  assert jnp.allclose(lc, expected, atol=1e-5)


def test_forward_not_implemented():
  n_dimension, n = 4, 10
  y = jr.normal(jr.PRNGKey(0), (n, n_dimension))

  init, _ = transform_layer(lambda: _build(n_dimension), "forward")
  with pytest.raises(NotImplementedError):
    init(jr.PRNGKey(1), y)
