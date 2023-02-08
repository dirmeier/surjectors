import chex
import distrax
import haiku as hk
from jax import numpy as jnp

from surjectors.bijectors.lu_linear import LULinear
from surjectors.surjectors.funnel import Funnel


class MLPFunnel(Funnel, hk.Module):
    """
    Multilayer perceptron funnel

    The MLPFunnel is an inference surjection
    """

    def __init__(self, n_keep, decoder, dtype=jnp.float32):
        self._r = LULinear(n_keep, False, dtype)
        self._w_prime = hk.Linear(n_keep, True)
        self._decoder = decoder
        super().__init__(n_keep, decoder, None, None, "inference_surjector")

    def split_input(self, array):
        """Split an array"""
        spl = jnp.split(array, [self.n_keep], axis=-1)
        return spl

    def inverse_and_likelihood_contribution(self, y: chex.Array, **kwargs):
        y_plus, y_minus = self.split_input(y)
        z, jac_det = self._r.inverse_and_likelihood_contribution(y_plus)
        z += self._w_prime(y_minus)
        lp = self._decode(z).log_prob(y_minus)
        return z, lp + jac_det

    def _decode(self, array):
        mu, log_scale = self._decoder(array)
        distr = distrax.MultivariateNormalDiag(mu, jnp.exp(log_scale))
        return distr

    def forward_and_likelihood_contribution(self, z: chex.Array, **kwargs):
        pass
