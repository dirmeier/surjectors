from jax import numpy as jnp

from surjectors.surjectors.affine_coupling_funnel import Funnel


class Slice(Funnel):
    def __init__(self, n_keep, decoder, encoder=None, kind="inference_surjection"):
        super().__init__(n_keep, decoder, encoder, kind)

    def inverse_and_likelihood_contribution(self, y):
        z, y_minus = self.split_input(y)
        lc = self.decoder.log_prob(y_minus, context=z)
        return z, lc

    def forward_and_likelihood_contribution(self, z):
        y_minus = self.decoder.sample(context=z)
        y = jnp.concatenate([z, y_minus], axis=-1)
        return y
