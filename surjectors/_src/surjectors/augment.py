import haiku as hk
from jax import numpy as jnp

from surjectors._src.surjectors.surjector import Surjector


class Augment(Surjector):
    """Augment generative funnel."""

    def __init__(self, n_keep, encoder):
        """Construct an augmentation layer.

        Args:
            n_keep: number of dimensions to keep
            encoder: encoder callable
        """
        self.n_keep = n_keep
        self.encoder = encoder

    def _split_input(self, array):
        spl = jnp.split(array, [self.n_keep], axis=-1)
        return spl

    def _inverse_and_likelihood_contribution(self, y, x=None, **kwargs):
        z_plus = y_condition = y
        if x is not None:
            y_condition = jnp.concatenate([y_condition, x], axis=-1)
        z_minus, lc = self.encoder(y_condition).sample_and_log_prob(
            seed=hk.next_rng_key()
        )
        z = jnp.concatenate([z_plus, z_minus], axis=-1)
        return z, -lc

    def _forward_and_likelihood_contribution(self, z, x=None, **kwargs):
        z_plus, z_minus = self._split_input(z)
        y_condition = y = z_plus
        if x is not None:
            y_condition = jnp.concatenate([y_condition, x], axis=-1)
        lc = self.encoder(y_condition).log_prob(z_minus)
        return y, -lc
