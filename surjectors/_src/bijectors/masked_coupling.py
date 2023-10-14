from typing import Callable, Optional

import distrax
from distrax._src.utils import math
from jax import numpy as jnp

from surjectors._src.bijectors.bijector import Bijector
from surjectors._src.distributions.transformed_distribution import Array


# pylint: disable=too-many-arguments, arguments-renamed
class MaskedCoupling(Bijector, distrax.MaskedCoupling):
    """A masked coupling layer.

    References:
        .. [1] Dinh, Laurent, et al. "Density estimation using RealNVP".
            International Conference on Learning Representations, 2017.

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
    """

    def __init__(
        self,
        mask: Array,
        conditioner: Callable,
        bijector_fn: Callable,
        event_ndims: Optional[int] = None,
        inner_event_ndims: int = 0,
    ):
        """Construct a masked coupling layer.

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
        """
        super().__init__(
            mask, conditioner, bijector_fn, event_ndims, inner_event_ndims
        )

    def _forward_and_likelihood_contribution(self, z, x=None, **kwargs):
        self._check_forward_input_shape(z)
        masked_z = jnp.where(self._event_mask, z, 0.0)
        if x is not None:
            masked_z = jnp.concatenate([masked_z, x], axis=-1)
        params = self._conditioner(masked_z)
        y0, log_d = self._inner_bijector(params).forward_and_log_det(z)
        y = jnp.where(self._event_mask, z, y0)
        logdet = math.sum_last(
            jnp.where(self._mask, 0.0, log_d),
            self._event_ndims - self._inner_event_ndims,
        )
        return y, logdet

    def _inverse_and_likelihood_contribution(self, y, x=None, **kwargs):
        self._check_inverse_input_shape(y)
        masked_y = jnp.where(self._event_mask, y, 0.0)
        if x is not None:
            masked_y = jnp.concatenate([masked_y, x], axis=-1)
        params = self._conditioner(masked_y)
        z0, log_d = self._inner_bijector(params).inverse_and_log_det(y)
        z = jnp.where(self._event_mask, y, z0)
        logdet = math.sum_last(
            jnp.where(self._mask, 0.0, log_d),
            self._event_ndims - self._inner_event_ndims,
        )
        return z, logdet
