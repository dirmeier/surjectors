from typing import Callable

import chex
import distrax
import haiku as hk
from jax import numpy as jnp

from surjectors._src.bijectors.lu_linear import LULinear


from surjectors._src.surjectors.surjector import Surjector


class MLPInferenceFunnel(Surjector, hk.Module):
    """
    A multilayer perceptron inference funnel.

    Examples:

        >>> from surjectors import MLPInferenceFunnel
        >>> from surjectors.nn import make_mlp
        >>>
        >>> def decoder_fn(n_dim):
        >>>     def _fn(z):
        >>>         params = make_mlp([4, 4, n_dim * 2])(y)
        >>>         mu, log_scale = jnp.split(params, 2, -1)
        >>>         return distrax.Independent(distrax.Normal(mu, jnp.exp(log_scale)))
        >>>     return _fn
        >>>
        >>> decoder = decoder_fn(5)
        >>> a = MLPInferenceFunnel(10, decoder)
    """

    def __init__(self, n_keep: int, decoder: Callable):
        """
        Constructs a MLPInferenceFunnel layer.

        Args:
            n_keep: number of dimensions to keep
            decoder: a conditional probability function
            dtype: parameter dtype
        """

        super().__init__()
        self._r = LULinear(n_keep, False)
        self._w_prime = hk.Linear(n_keep, True)
        self.decoder = decoder
        self.n_keep = n_keep

    def split_input(self, array):
        """Split an array into halves"""
        spl = jnp.split(array, [self.n_keep], axis=-1)
        return spl

    def inverse_and_likelihood_contribution(self, y: chex.Array, **kwargs):
        y_plus, y_minus = self.split_input(y)
        z, jac_det = self._r.inverse_and_likelihood_contribution(y_plus)
        z += self._w_prime(y_minus)
        lp = self._decode(z).log_prob(y_minus)
        return z, lp + jac_det

    def _decode(self, array):
        mu, log_scale = self.decoder(array)
        distr = distrax.MultivariateNormalDiag(mu, jnp.exp(log_scale))
        return distr

    def forward_and_likelihood_contribution(self, z: chex.Array, **kwargs):
        pass
