import warnings

import distrax
from chex import Array

from surjectors._src.bijectors.masked_autoregressive import MaskedAutoregressive
from surjectors._src.surjectors.affine_masked_autoregressive_inference_funnel import (  # noqa: E501
    AffineMaskedAutoregressiveInferenceFunnel,
)


# pylint: disable=too-many-arguments, arguments-renamed
class RationalQuadraticSplineMaskedAutoregressiveInferenceFunnel(
    AffineMaskedAutoregressiveInferenceFunnel
):
    """
    A masked autoregressive inference funnel that uses a rational quatratic
    spline as a transformation.

    Examples:

        >>> import distrax
        >>> from jax import numpy as jnp
        >>> from surjectors import RationalQuadraticSplineMaskedCouplingInferenceFunnel
        >>> from surjectors.nn import make_mlp
        >>>
        >>> def decoder_fn(n_dim):
        >>>     def _fn(z):
        >>>         params = make_mlp([4, 4, n_dim * 2])(z)
        >>>         mu, log_scale = jnp.split(params, 2, -1)
        >>>         return distrax.Independent(distrax.Normal(mu, jnp.exp(log_scale)))
        >>>     return _fn
        >>>
        >>> layer = RationalQuadraticSplineMaskedAutoregressiveInferenceFunnel(
        >>>     n_keep=10,
        >>>     decoder=decoder_fn(10),
        >>>     conditioner=MADE(10, [8, 8], 2),
        >>> )
    """

    def __init__(self, n_keep, decoder, conditioner, range_min, range_max):
        warnings.warn("class has not been tested. use at own risk")
        super().__init__(n_keep, decoder, conditioner)
        self.range_min = range_min
        self.range_max = range_max

    def _inner_bijector(self):
        def _bijector_fn(params: Array):
            return distrax.RationalQuadraticSpline(
                params, self.range_min, self.range_max
            )

        return MaskedAutoregressive(self._conditioner, _bijector_fn)
