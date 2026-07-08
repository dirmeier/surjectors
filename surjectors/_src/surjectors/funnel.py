from jax import numpy as jnp

from surjectors._src.surjectors.surjector import Surjector


class Funnel(Surjector):
  """Base class for dimensionality-changing surjectors.

  A funnel keeps ``n_keep`` dimensions and transforms the remainder. This
  base owns the two operations every funnel repeats: splitting an array at
  ``n_keep`` and conditioning a latent on an optional context.

  Args:
      n_keep: number of dimensions to keep
  """

  def __init__(self, n_keep: int):
    super().__init__()
    self.n_keep = n_keep

  def _split_input(self, array):
    """Split an array into the kept and transformed parts.

    Args:
        array: array to split along its last axis

    Returns:
        list of two arrays; the first has ``n_keep`` columns
    """
    return jnp.split(array, [self.n_keep], axis=-1)

  def _condition(self, array, x):
    """Concatenate an optional context onto an array.

    Args:
        array: array to be conditioned
        x: optional context; if ``None`` the array is returned unchanged

    Returns:
        ``array`` when ``x`` is ``None``, else ``array`` concatenated with
        ``x`` on the last axis
    """
    if x is not None:
      return jnp.concatenate([array, x], axis=-1)
    return array
