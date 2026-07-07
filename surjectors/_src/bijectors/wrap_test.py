import distrax
import pytest
from jax import numpy as jnp

from surjectors._src.bijectors.wrap import _BijectorAdapter, as_transform


def test_as_transform_passes_native_through():
  native = as_transform(distrax.ScalarAffine(0.0, 1.0))
  assert isinstance(native, _BijectorAdapter)


def test_adapter_raises_when_context_supplied():
  adapter = _BijectorAdapter(distrax.ScalarAffine(0.0, 1.0))
  with pytest.raises(ValueError, match="cannot condition"):
    adapter._forward_and_likelihood_contribution(jnp.zeros(3), x=jnp.ones(3))
  with pytest.raises(ValueError, match="cannot condition"):
    adapter._inverse_and_likelihood_contribution(jnp.zeros(3), x=jnp.ones(3))
