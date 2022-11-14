from jax import numpy as jnp
import haiku as hk
from surjectors.surjectors.funnel import Funnel


class Slice(Funnel):
    def __init__(
        self, n_keep, decoder, encoder=None, kind="inference_surjector"
    ):
        super().__init__(n_keep, decoder, encoder, None, kind)

    def split_input(self, input):
        spl = jnp.split(input, [self.n_keep], axis=-1)
        return spl

    def inverse_and_likelihood_contribution(self, y, x = None):
        z, y_minus = self.split_input(y)
        z_condition = z
        if x is not None:
            z_condition = jnp.concatenate([z, x], axis=-1)
        lc = self.decoder(z_condition).log_prob(y_minus)
        return z, lc

    def forward_and_likelihood_contribution(self, z, x=None):
        z_condition = z
        if x is not None:
            z_condition = jnp.concatenate([z, x], axis=-1)
        y_minus, lc = self.decoder(z_condition).sample_and_log_prob(
            seed=hk.next_rng_key()
        )
        y = jnp.concatenate([z, y_minus], axis=-1)
        return y, lc

    def forward(self, z, x=None):
        y, _ = self.forward_and_likelihood_contribution(z, x)
        return y
