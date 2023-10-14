from typing import Callable

import distrax
from chex import Array
from jax import numpy as jnp

from surjectors._src.conditioners.nn.made import MADE
from surjectors._src.surjectors.masked_autoregressive_inference_funnel import (
    MaskedAutoregressiveInferenceFunnel,
)
from surjectors.util import unstack


class AffineMaskedAutoregressiveInferenceFunnel(
    MaskedAutoregressiveInferenceFunnel
):
    """A masked affine autoregressive funnel layer.

    The AffineMaskedAutoregressiveInferenceFunnel is an autoregressive funnel,
    i.e., dimensionality reducing transformation, that uses an affine
    transformation from data to latent space using a masking mechanism as in
    MaskedAutoegressive.

    References:
        .. [1] Klein, Samuel, et al. "Funnels: Exact maximum likelihood
            with dimensionality reduction". Workshop on Bayesian Deep Learning,
            Advances in Neural Information Processing Systems, 2021.
        .. [2] Papamakarios, George, et al. "Masked Autoregressive Flow for
            Density Estimation". Advances in Neural Information Processing
            Systems, 2017.

    Examples:
        >>> import distrax
        >>> from surjectors import AffineMaskedAutoregressiveInferenceFunnel
        >>> from surjectors.nn import MADE, make_mlp
        >>> from surjectors.util import unstack
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
        >>> layer = AffineMaskedAutoregressiveInferenceFunnel(
        >>>     n_keep=10,
        >>>     decoder=decoder_fn(10),
        >>>     conditioner=MADE(10, [8, 8], 2),
        >>> )
    """

    def __init__(self, n_keep: int, decoder: Callable, conditioner: MADE):
        """Constructs a AffineMaskedAutoregressiveInferenceFunnel layer.

        Args:
            n_keep: number of dimensions to keep
            decoder: a callable that returns a conditional probabiltiy
                distribution when called
            conditioner: a MADE neural network
        """

        def bijector_fn(params: Array):
            shift, log_scale = unstack(params, axis=-1)
            return distrax.ScalarAffine(shift, jnp.exp(log_scale))

        super().__init__(n_keep, decoder, conditioner, bijector_fn)
