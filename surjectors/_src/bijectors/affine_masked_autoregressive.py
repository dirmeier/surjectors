import distrax
from jax import numpy as jnp

from surjectors._src.bijectors.masked_autoregressive import MaskedAutoregressive
from surjectors._src.conditioners.nn.made import MADE
from surjectors.util import unstack


# pylint: disable=too-many-arguments,arguments-renamed
class AffineMaskedAutoregressive(MaskedAutoregressive):
    """An affine masked autoregressive layer.

    Args:
        conditioner: a MADE network
        event_ndims: the number of array dimensions the bijector operates on
        inner_event_ndims: tthe number of array dimensions the bijector
            operates on

    References:
        .. [1] Papamakarios, George, et al. "Masked Autoregressive Flow for
            Density Estimation". Advances in Neural Information Processing
            Systems, 2017.

    Examples:
        >>> import distrax
        >>> from surjectors import AffineMaskedAutoregressive
        >>>
        >>> layer = AffineMaskedAutoregressive(
        >>>     conditioner=MADE(10, [8, 8], 2),
        >>> )
    """

    def __init__(
        self,
        conditioner: MADE,
        event_ndims: int = 1,
        inner_event_ndims: int = 0,
    ):
        def bijector_fn(params):
            means, log_scales = unstack(params, -1)
            return distrax.ScalarAffine(means, jnp.exp(log_scales))

        super().__init__(
            conditioner, bijector_fn, event_ndims, inner_event_ndims
        )
