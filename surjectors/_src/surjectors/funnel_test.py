# noqa
# pylint: skip-file

from jax import numpy as jnp

from surjectors._src.surjectors.funnel import Funnel


class _ConcreteFunnel(Funnel):
  def _inverse_and_likelihood_contribution(self, y, x=None, **kwargs):
    raise NotImplementedError

  def _forward_and_likelihood_contribution(self, z, x=None, **kwargs):
    raise NotImplementedError


def test_split_input_splits_at_n_keep():
  funnel = _ConcreteFunnel(3)
  array = jnp.arange(10).reshape(1, 10)
  left, right = funnel._split_input(array)
  assert left.shape == (1, 3)
  assert right.shape == (1, 7)


def test_condition_without_context_returns_input_unchanged():
  funnel = _ConcreteFunnel(3)
  array = jnp.ones((2, 3))
  assert jnp.array_equal(funnel._condition(array, None), array)


def test_condition_with_context_concatenates_on_last_axis():
  funnel = _ConcreteFunnel(3)
  array = jnp.ones((2, 3))
  x = jnp.zeros((2, 4))
  out = funnel._condition(array, x)
  assert out.shape == (2, 7)
  assert jnp.array_equal(out[:, :3], array)
  assert jnp.array_equal(out[:, 3:], x)
