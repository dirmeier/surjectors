from jax import numpy as jnp

from surjectors.surjectors.surjector import Surjector


class Funnel(Surjector):
    def __init__(self, n_keep, decoder, encoder=None, kind="inference_surjection"):
        super().__init__(n_keep, decoder, encoder, kind)

    def split_input(self, input):
        split_proportions = (self.n_keep, input.shape[-1] - self.n_keep)
        return jnp.split(input, split_proportions, axis=-1)

    def inverse_and_likelihood_contribution(self, y):
        z, y_minus = self.split_input(y)
        lc = self.decoder.log_prob(y_minus, context=z)
        return z, lc

    def forward_and_likelihood_contribution(self, z):
        y_minus = self.decoder.sample(context=z)
        y = jnp.concatenate([z, y_minus], axis=-1)
        return y
