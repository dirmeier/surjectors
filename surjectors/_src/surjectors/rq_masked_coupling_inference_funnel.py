import distrax
from chex import Array

# pylint: disable=too-many-arguments, arguments-renamed
from surjectors._src.surjectors.masked_coupling_inference_funnel import (
    MaskedCouplingInferenceFunnel,
)


class RationalQuadraticSplineMaskedCouplingInferenceFunnel(
    MaskedCouplingInferenceFunnel
):
    """A masked coupling inference funnel that uses a rational quatratic spline.

    References:
        .. [1] Klein, Samuel, et al. "Funnels: Exact maximum likelihood
            with dimensionality reduction". Workshop on Bayesian Deep Learning,
            Advances in Neural Information Processing Systems, 2021.
        .. [2] Durkan, Conor, et al. "Neural Spline Flows".
            Advances in Neural Information Processing Systems, 2019.
        .. [3] Dinh, Laurent, et al. "Density estimation using RealNVP".
            International Conference on Learning Representations, 2017.

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
        >>> layer = RationalQuadraticSplineMaskedCouplingInferenceFunnel(
        >>>     n_keep=10,
        >>>     decoder=decoder_fn(10),
        >>>     conditioner=make_mlp([4, 4, 10 * 2])(z),
        >>> )
    """

    def __init__(self, n_keep, decoder, conditioner, range_min, range_max):
        """Construct a RationalQuadraticSplineMaskedCouplingInferenceFunnel.

        Args:
            n_keep: number of dimensions to keep
            decoder: a callable that returns a conditional probabiltiy
                distribution when called
            conditioner: a conditioning neural network
            range_min: minimum range of the spline
            range_max: maximum range of the spline
        """
        self.range_min = range_min
        self.range_max = range_max

        def _bijector_fn(params: Array):
            return distrax.RationalQuadraticSpline(
                params, self.range_min, self.range_max
            )

        super().__init__(n_keep, decoder, conditioner, _bijector_fn)
