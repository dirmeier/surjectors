from typing import Callable, Optional

import distrax
from jax import numpy as jnp

from surjectors._src.bijectors.masked_coupling import MaskedCoupling
from surjectors._src.distributions.transformed_distribution import Array


# pylint: disable=too-many-arguments, arguments-renamed,too-many-ancestors
class AffineMaskedCoupling(MaskedCoupling):
    """An affine masked coupling layer.

    Args:
        mask: a boolean mask of length n_dim. A value
            of True indicates that the corresponding input remains unchanged
        conditioner: a function that computes the parameters of the inner
            bijector
        event_ndims: the number of array dimensions the bijector operates on
        inner_event_ndims: the number of array dimensions the inner bijector
            operates on

    References:
        .. [1] Dinh, Laurent, et al. "Density estimation using RealNVP".
            International Conference on Learning Representations, 2017.

    Examples:
        >>> import distrax
        >>> from surjectors import AffineMaskedCoupling
        >>> from surjectors.nn import make_mlp
        >>> from surjectors.util import make_alternating_binary_mask
        >>>
        >>> layer = MaskedCoupling(
        >>>     mask=make_alternating_binary_mask(10, True),
        >>>     conditioner=make_mlp([8, 8, 10 * 2]),
        >>> )
    """

    def __init__(
        self,
        mask: Array,
        conditioner: Callable,
        event_ndims: Optional[int] = None,
        inner_event_ndims: int = 0,
    ):
        def _bijector_fn(params):
            means, log_scales = jnp.split(params, 2, -1)
            return distrax.ScalarAffine(means, jnp.exp(log_scales))

        super().__init__(
            mask, conditioner, _bijector_fn, event_ndims, inner_event_ndims
        )
