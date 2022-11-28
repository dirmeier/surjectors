import distrax
from chex import Array
from jax import numpy as jnp
import haiku as hk
from surjectors.surjectors.funnel import Funnel


class Augment(Funnel):
    def __init__(self, n_keep, encoder):
        super().__init__(n_keep, None, None, encoder, "generative_surjector")

    def split_input(self, input):
        spl = jnp.split(input, [self.n_keep], axis=-1)
        return spl

    def inverse_and_likelihood_contribution(self, y, x: Array = None):
        z_plus = y_condition = y
        if x is not None:
            y_condition = jnp.concatenate([y_condition, x], axis=-1)
        z_minus, lc = self.encoder(y_condition).sample_and_log_prob(seed=hk.next_rng_key())
        z = jnp.concatenate([z_plus, z_minus], axis=-1)
        return z, -lc

    def forward_and_likelihood_contribution(self, z, x=None):
        z_plus, z_minus = self.split_input(z)
        y_condition = y = z_plus
        if x is not None:
            y_condition = jnp.concatenate([y_condition, x], axis=-1)
        lc = self.encoder(y_condition).log_prob(z_minus)
        return y, -lc

    def forward(self, z, x=None):
        y, _ = self.forward_and_likelihood_contribution(z, x)
        return y
