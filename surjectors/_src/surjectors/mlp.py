from typing import Callable

import haiku as hk
from jax import numpy as jnp

from surjectors._src.bijectors.lu_linear import LULinear
from surjectors._src.surjectors.surjector import Surjector


class MLPInferenceFunnel(Surjector, hk.Module):
    """A multilayer perceptron inference funnel.

    References:
        .. [1] Klein, Samuel, et al. "Funnels: Exact maximum likelihood
            with dimensionality reduction". Workshop on Bayesian Deep Learning,
            Advances in Neural Information Processing Systems, 2021.

    Examples:
        >>> import distrax
        >>> from surjectors import MLPInferenceFunnel
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
        >>> decoder = decoder_fn(5)
        >>> a = MLPInferenceFunnel(10, decoder)
    """

    def __init__(self, n_keep: int, decoder: Callable):
        """Constructs a MLPInferenceFunnel layer.

        Args:
            n_keep: number of dimensions to keep
            decoder: a conditional probability function
        """
        super().__init__()
        self._r = LULinear(n_keep, False)
        self._w_prime = hk.Linear(n_keep, True)
        self.decoder = decoder
        self.n_keep = n_keep

    def _split_input(self, array):
        spl = jnp.split(array, [self.n_keep], axis=-1)
        return spl

    def _inverse_and_likelihood_contribution(self, y, x=None, **kwargs):
        y_plus, y_minus = self._split_input(y)
        z, jac_det = self._r.inverse_and_likelihood_contribution(y_plus)
        z += self._w_prime(y_minus)
        lp = self.decoder(z).log_prob(y_minus)
        return z, lp + jac_det

    def _forward_and_likelihood_contribution(self, z, x=None, **kwargs):
        raise NotImplementedError()
