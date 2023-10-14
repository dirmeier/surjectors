from typing import Callable

import haiku as hk
from jax import numpy as jnp

from surjectors._src.surjectors.surjector import Surjector


class Slice(Surjector):
    """A slice funnel.

    References:
        .. [1] Nielsen, Didrik, et al. "SurVAE Flows: Surjections to Bridge the
            Gap between VAEs and Flows". Advances in Neural Information
            Processing Systems, 2020.

    Examples:
        >>> import distrax
        >>> from surjectors import Slice
        >>> from surjectors.nn import make_mlp
        >>>
        >>> def decoder_fn(n_dim):
        >>>     def _fn(z):
        >>>         params = make_mlp([4, 4, n_dim * 2])(z)
        >>>         mu, log_scale = jnp.split(params, 2, -1)
        >>>         return distrax.Independent(
        >>>             distrax.Normal(mu, jnp.exp(log_scale))
        >>>         )
        >>>     return _fn
        >>>
        >>> layer = Slice(10, decoder_fn(10))
    """

    def __init__(self, n_keep: int, decoder: Callable):
        """Constructs a slice layer.

        Args:
            n_keep: number if dimensions to keep
            decoder: callable
        """
        self.n_keep = n_keep
        self.decoder = decoder

    def _split_input(self, array):
        spl = jnp.split(array, [self.n_keep], axis=-1)
        return spl

    def _inverse_and_likelihood_contribution(self, y, x=None, **kwargs):
        z, y_minus = self._split_input(y)
        z_condition = z
        if x is not None:
            z_condition = jnp.concatenate([z, x], axis=-1)
        lc = self.decoder(z_condition).log_prob(y_minus)
        return z, lc

    def _forward_and_likelihood_contribution(self, z, x=None, **kwargs):
        z_condition = z
        if x is not None:
            z_condition = jnp.concatenate([z, x], axis=-1)
        y_minus, lc = self.decoder(z_condition).sample_and_log_prob(
            seed=hk.next_rng_key()
        )
        y = jnp.concatenate([z, y_minus], axis=-1)
        return y, lc
