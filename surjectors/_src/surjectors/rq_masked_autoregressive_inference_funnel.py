import warnings
from typing import Callable

import distrax
from chex import Array

from surjectors._src.bijectors.masked_autoregressive import MaskedAutoregressive
from surjectors._src.conditioners.nn.made import MADE
from surjectors._src.surjectors.masked_autoregressive_inference_funnel import (
    MaskedAutoregressiveInferenceFunnel,
)


# pylint: disable=too-many-arguments, arguments-renamed
class RationalQuadraticSplineMaskedAutoregressiveInferenceFunnel(
    MaskedAutoregressiveInferenceFunnel
):
    """A masked autoregressive inference funnel that uses RQ-NSFs.

    References:
        .. [1] Klein, Samuel, et al. "Funnels: Exact maximum likelihood
            with dimensionality reduction". Workshop on Bayesian Deep Learning,
            Advances in Neural Information Processing Systems, 2021.
        .. [2] Durkan, Conor, et al. "Neural Spline Flows".
            Advances in Neural Information Processing Systems, 2019.
        .. [3] Papamakarios, George, et al. "Masked Autoregressive Flow for
            Density Estimation". Advances in Neural Information Processing
            Systems, 2017.

    Examples:
        >>> import distrax
        >>> from jax import numpy as jnp
        >>> from surjectors import \
        >>>     RationalQuadraticSplineMaskedCouplingInferenceFunnel
        >>> from surjectors.nn import make_mlp
        >>>
        >>> def decoder_fn(n_dim):
        >>>     def _fn(z):
        >>>         params = make_mlp([4, 4, n_dim * 2])(z)
        >>>         mu, log_scale = jnp.split(params, 2, -1)
        >>>         return distrax.Independent(
        >>>             distrax.Normal(mu, jnp.exp(log_scale))
        >>>         )
        >>>     return _fn
        >>>
        >>> layer = RationalQuadraticSplineMaskedAutoregressiveInferenceFunnel(
        >>>     n_keep=10,
        >>>     decoder=decoder_fn(10),
        >>>     conditioner=MADE(10, [8, 8], 2),
        >>> )
    """

    def __init__(
        self,
        n_keep: int,
        decoder: Callable,
        conditioner: MADE,
        range_min: float,
        range_max: float,
    ):
        """Constructs a RQ-NSF inference funnel.

        Args:
            n_keep: number of dimensions to keep
            decoder: a callable that returns a conditional probabiltiy
                distribution when called
            conditioner: a conditioning neural network
            range_min: minimum range of the spline
            range_max: maximum range of the spline
        """
        warnings.warn("class has not been tested. use at own risk")
        self.range_min = range_min
        self.range_max = range_max

        def _inner_bijector(self):
            def _bijector_fn(params: Array):
                return distrax.RationalQuadraticSpline(
                    params, self.range_min, self.range_max
                )

            return MaskedAutoregressive(self.conditioner, _bijector_fn)

        super().__init__(n_keep, decoder, conditioner, _inner_bijector)
