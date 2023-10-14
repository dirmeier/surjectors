from typing import Callable

import distrax
from chex import Array
from jax import numpy as jnp

from surjectors._src.surjectors.masked_coupling_inference_funnel import (
    MaskedCouplingInferenceFunnel,
)


class AffineMaskedCouplingInferenceFunnel(MaskedCouplingInferenceFunnel):
    """A masked coupling inference funnel that uses an affine transformation.

    References:
        .. [1] Klein, Samuel, et al. "Funnels: Exact maximum likelihood
            with dimensionality reduction". Workshop on Bayesian Deep Learning,
            Advances in Neural Information Processing Systems, 2021.
        .. [2] Dinh, Laurent, et al. "Density estimation using RealNVP".
            International Conference on Learning Representations, 2017.

    Examples:
        >>> import distrax
        >>> from jax import numpy as jnp
        >>> from surjectors import AffineMaskedCouplingInferenceFunnel
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
        >>> layer = AffineMaskedCouplingInferenceFunnel(
        >>>     n_keep=10,
        >>>     decoder=decoder_fn(10),
        >>>     conditioner=make_mlp([4, 4, 10 * 2])(z),
        >>> )
    """

    def __init__(self, n_keep: int, decoder: Callable, conditioner: Callable):
        """Constructs a AffineMaskedCouplingInferenceFunnel layer.

        Args:
            n_keep: number of dimensions to keep
            decoder: a callable that returns a conditional probabiltiy
                distribution when called
            conditioner: a conditioning neural network
        """

        def bijector_fn(params: Array):
            shift, log_scale = jnp.split(params, 2, axis=-1)
            return distrax.ScalarAffine(shift, jnp.exp(log_scale))

        super().__init__(n_keep, decoder, conditioner, bijector_fn)
