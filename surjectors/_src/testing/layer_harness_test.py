import distrax
import pytest
from jax import numpy as jnp
from jax import random as jr

from surjectors._src.bijectors.masked_coupling import MaskedCoupling
from surjectors._src.conditioners.mlp import make_mlp
from surjectors._src.testing.layer_harness import transform_layer
from surjectors.util import make_alternating_binary_mask


def _build():
  def bij(params):
    shift, log_scale = jnp.split(params, 2, -1)
    return distrax.ScalarAffine(shift, jnp.exp(log_scale))

  mask = make_alternating_binary_mask(4, True)
  return MaskedCoupling(mask, make_mlp([8, 8, 8]), bij)


def test_masked_coupling_round_trips():
  y = jr.normal(jr.PRNGKey(0), (16, 4))
  inv_init, inv_apply = transform_layer(_build, "inverse")
  params = inv_init(jr.PRNGKey(1), y)
  z, ldj_inv = inv_apply(params, y)
  assert z.shape == y.shape
  assert ldj_inv.shape == (16,)

  _, fwd_apply = transform_layer(_build, "forward")
  y_hat, ldj_fwd = fwd_apply(params, z)
  assert y_hat.shape == y.shape
  assert jnp.allclose(y_hat, y, atol=1e-4)
  assert jnp.allclose(ldj_inv, -ldj_fwd, atol=1e-4)


def test_transform_layer_rejects_bad_direction():
  with pytest.raises(ValueError, match="direction"):
    transform_layer(_build, "sideways")
