import chex
from jax import numpy as jnp

from surjectors.surjectors.surjector import Surjector


class AffineCouplingFunnel(Surjector):
    def __init__(self, n_keep, decoder, transform, encoder, kind="inference_surjection"):
        super().__init__(n_keep, decoder, encoder, kind)
        self._transform = transform

    def split_input(self, input):
        split_proportions = (self.n_keep, input.shape[-1] - self.n_keep)
        return jnp.split(input, split_proportions, axis=-1)

    def inverse_and_likelihood_contribution(self, y):
        y_plus, y_minus = self.split_input(y)
        chex.assert_equal_shape([y_plus, y_minus])
        z, jac_det = self._transform(y_plus, context=y_minus)
        lp = self.decoder.log_prob(y_minus, context=z)
        return z, lp + jac_det

    def forward_and_likelihood_contribution(self, z):
        raise NotImplementedError()
