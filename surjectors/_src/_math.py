"""Small array helpers."""

from jax import Array


def sum_last(x: Array, ndims: int) -> Array:
  """Sum an array over its trailing ``ndims`` axes.

  Args:
      x: array to reduce
      ndims: number of trailing axes to sum over. ``0`` returns ``x``
          unchanged.

  Returns:
      the array summed over its last ``ndims`` axes
  """
  return x.sum(axis=tuple(range(-ndims, 0)))
