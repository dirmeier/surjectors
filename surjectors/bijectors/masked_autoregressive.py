from typing import Callable

from distrax._src.utils import math
from jax import numpy as jnp

from surjectors.bijectors._bijector import _Bijector
from surjectors.conditioners import MADE
from surjectors.distributions.transformed_distribution import Array


# pylint: disable=too-many-arguments, arguments-renamed
class MaskedAutoregressive(_Bijector):
    """
    Masked autoregressive layer
    """

    def __init__(
        self,
        conditioner: MADE,
        bijector_fn: Callable,
        event_ndims: int = 1,
        inner_event_ndims: int = 0,
        dtype=jnp.float32,
    ):
        if event_ndims is not None and event_ndims < inner_event_ndims:
            raise ValueError(
                f"`event_ndims={event_ndims}` should be at least as"
                f" large as `inner_event_ndims={inner_event_ndims}`."
            )
        self._event_ndims = event_ndims
        self._inner_event_ndims = inner_event_ndims
        super().__init__(conditioner, bijector_fn, dtype)
        if not isinstance(conditioner, MADE):
            raise ValueError(
                "conditioner should be a MADE when used "
                "MaskedAutoregressive flow"
            )

    def forward_and_log_det(self, z, x: Array = None):
        y = jnp.zeros_like(z)
        for _ in jnp.arange(z.shape[-1]):
            params = self.conditioner(y, x)
            y, log_det = self._inner_bijector(params).forward_and_log_det(z)
        log_det = math.sum_last(
            log_det, self._event_ndims - self._inner_event_ndims
        )
        return y, log_det

    def inverse_and_log_det(self, y, x: Array = None):
        params = self.conditioner(y, x)
        z, log_det = self._inner_bijector(params).inverse_and_log_det(y)
        log_det = math.sum_last(
            log_det, self._event_ndims - self._inner_event_ndims
        )
        return z, log_det

    def forward(self, z: Array, x: Array = None) -> Array:
        y, _ = self.forward_and_log_det(z, x)
        return y

    def inverse_and_likelihood_contribution(self, y, x: Array = None, **kwargs):
        return self.inverse_and_log_det(y, x)

    def forward_and_likelihood_contribution(self, z, x: Array = None, **kwargs):
        return self.forward_and_log_det(z, x)
