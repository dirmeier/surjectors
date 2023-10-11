import haiku as hk
from chex import Array
from jax import numpy as jnp

from surjectors.surjectors.funnel import Funnel


class Slice(Funnel):
    """
    Slice funnel
    """

    def __init__(self, n_keep, decoder):
        super().__init__(n_keep, decoder, None, None, "inference_surjector")

    def split_input(self, array):
        """Split an array"""
        spl = jnp.split(array, [self.n_keep], axis=-1)
        return spl

    def inverse_and_likelihood_contribution(self, y, x: Array = None, **kwargs):
        z, y_minus = self.split_input(y)
        z_condition = z
        if x is not None:
            z_condition = jnp.concatenate([z, x], axis=-1)
        lc = self.decoder(z_condition).log_prob(y_minus)
        return z, lc

    def forward_and_likelihood_contribution(self, z, x: Array = None, **kwargs):
        z_condition = z
        if x is not None:
            z_condition = jnp.concatenate([z, x], axis=-1)
        y_minus, lc = self.decoder(z_condition).sample_and_log_prob(
            seed=hk.next_rng_key()
        )
        y = jnp.concatenate([z, y_minus], axis=-1)

        return y, lc

    def forward(self, z, x: Array = None):
        y, _ = self.forward_and_likelihood_contribution(z, x)
        return y
