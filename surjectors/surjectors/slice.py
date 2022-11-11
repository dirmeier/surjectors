from jax import numpy as jnp

from surjectors.surjectors.funnel import Funnel


class Slice(Funnel):
    def __init__(
        self, n_keep, decoder, encoder=None, kind="inference_surjector"
    ):
        super().__init__(n_keep, decoder, encoder, None, kind)

    def split_input(self, input):
        spl = jnp.split(input, [self.n_keep], axis=-1)
        return spl

    def inverse_and_likelihood_contribution(self, y):
        z, y_minus = self.split_input(y)
        lc = self.decoder(z).log_prob(y_minus)
        return z, lc

    def forward_and_likelihood_contribution(self, z):
        y_minus = self.decoder.sample(context=z)
        y = jnp.concatenate([z, y_minus], axis=-1)
        return y
