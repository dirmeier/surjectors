from typing import Optional, Tuple

import distrax
from distrax._src.utils import math
from jax import numpy as jnp

from surjectors.distributions.transformed_distribution import Array
from surjectors.surjectors.surjector import Surjector


# pylint: disable=too-many-arguments, arguments-renamed
class MaskedCoupling(distrax.MaskedCoupling, Surjector):
    """
    Masked coupling layer
    """

    def __init__(
        self,
        mask: Array,
        conditioner,
        bijector,
        event_ndims: Optional[int] = None,
        inner_event_ndims: int = 0,
    ):
        super().__init__(
            mask, conditioner, bijector, event_ndims, inner_event_ndims
        )

    def forward_and_log_det(
        self, z: Array, x: Array = None
    ) -> Tuple[Array, Array]:
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

    def forward(self, z: Array, x: Array = None) -> Array:
        y, _ = self.forward_and_log_det(z, x)
        return y

    def inverse_and_log_det(
        self, y: Array, x: Array = None
    ) -> Tuple[Array, Array]:
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

    def inverse_and_likelihood_contribution(self, y, x: Array = None, **kwargs):
        return self.inverse_and_log_det(y, x)

    def forward_and_likelihood_contribution(self, z, x: Array = None, **kwargs):
        return self.forward_and_log_det(z, x)
