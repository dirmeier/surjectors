from collections.abc import Callable

from jax import numpy as jnp

from surjectors._src._math import sum_last
from surjectors._src.bijectors.bijector import Bijector
from surjectors._src.distributions.transformed_distribution import Array


# ruff: noqa: PLR0913
class MaskedCoupling(Bijector):
  """A masked coupling layer.

  Args:
      mask: a boolean mask of length n_dim. A value
          of True indicates that the corresponding input remains unchanged
      conditioner: a function that computes the parameters of the inner
          bijector
      bijector_fn: a callable that returns the inner bijector that will be
          used to transform the input
      event_ndims: the number of array dimensions the bijector operates on
      inner_event_ndims: the number of array dimensions the inner bijector
          operates on

  Examples:
      >>> import distrax
      >>> from surjectors import MaskedCoupling
      >>> from surjectors.nn import make_mlp
      >>> from surjectors.util import make_alternating_binary_mask
      >>>
      >>> def bijector_fn(params):
      >>>     means, log_scales = jnp.split(params, 2, -1)
      >>>     return distrax.ScalarAffine(means, jnp.exp(log_scales))
      >>>
      >>> layer = MaskedCoupling(
      >>>     mask=make_alternating_binary_mask(10, True),
      >>>     bijector_fn=bijector_fn,
      >>>     conditioner=make_mlp([8, 8, 10 * 2]),
      >>> )

  References:
      .. [1] Dinh, Laurent, et al. "Density estimation using RealNVP".
          International Conference on Learning Representations, 2017.
  """

  def __init__(
    self,
    mask: Array,
    conditioner: Callable,
    bijector_fn: Callable,
    event_ndims: int | None = None,
    inner_event_ndims: int = 0,
  ):
    if mask.dtype != bool:
      raise ValueError(
        f"'mask' must have boolean values; got dtype '{mask.dtype}'."
      )
    if event_ndims is not None and event_ndims < inner_event_ndims:
      raise ValueError(
        f"'event_ndims={event_ndims}' should be at least as large as"
        f" 'inner_event_ndims={inner_event_ndims}'."
      )
    self._mask = mask
    self._event_mask = jnp.reshape(mask, mask.shape + (1,) * inner_event_ndims)
    self._conditioner = conditioner
    self._inner_bijector = bijector_fn
    self._inner_event_ndims = inner_event_ndims
    if event_ndims is None:
      self._event_ndims = mask.ndim + inner_event_ndims
    else:
      self._event_ndims = event_ndims

  def _forward_and_likelihood_contribution(self, z, x=None, **kwargs):
    masked_z = jnp.where(self._event_mask, z, 0.0)
    if x is not None:
      masked_z = jnp.concatenate([masked_z, x], axis=-1)
    params = self._conditioner(masked_z)
    y0, log_d = self._inner_bijector(params).forward_and_log_det(z)
    y = jnp.where(self._event_mask, z, y0)
    logdet = sum_last(
      jnp.where(self._mask, 0.0, log_d),
      self._event_ndims - self._inner_event_ndims,
    )
    return y, logdet

  def _inverse_and_likelihood_contribution(self, y, x=None, **kwargs):
    masked_y = jnp.where(self._event_mask, y, 0.0)
    if x is not None:
      masked_y = jnp.concatenate([masked_y, x], axis=-1)
    params = self._conditioner(masked_y)
    z0, log_d = self._inner_bijector(params).inverse_and_log_det(y)
    z = jnp.where(self._event_mask, y, z0)
    logdet = sum_last(
      jnp.where(self._mask, 0.0, log_d),
      self._event_ndims - self._inner_event_ndims,
    )
    return z, logdet
