import distrax
import haiku as hk
from jax import numpy as jnp

from surjectors.surjectors.affine_masked_coupling_inference_funnel import Funnel
from surjectors.surjectors.lu_linear import LULinear


class MLP(Funnel, hk.Module):
    def __init__(self, n_keep, decoder, dtype=jnp.float32):
        self._r = LULinear(n_keep, dtype, with_bias=False)
        self._w_prime = hk.Linear(n_keep, with_bias=True)

        self._decoder = decoder
        super().__init__(n_keep, decoder)

    def inverse_and_likelihood_contribution(self, y):
        y_plus, y_minus = self.split_input(y)
        z, jac_det = self._r.inverse_and_likelihood_contribution(y_plus)
        z += self._w_prime(y_minus)
        lp = self._decode(z).log_prob(y_minus)
        return z, lp + jac_det

    def _decode(self, array):
        mu, log_scale = self._decoder(array)
        distr = distrax.MultivariateNormalDiag(mu, jnp.exp(log_scale))
        return distr

    def forward_and_likelihood_contribution(self, z):
        pass
