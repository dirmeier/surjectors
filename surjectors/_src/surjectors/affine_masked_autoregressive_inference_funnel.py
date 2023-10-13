from typing import Callable

import distrax
import haiku as hk
from chex import Array
from jax import numpy as jnp

from surjectors._src.bijectors.masked_autoregressive import MaskedAutoregressive
from surjectors._src.surjectors.surjector import Surjector

from surjectors.util import unstack


class AffineMaskedAutoregressiveInferenceFunnel(Surjector):
    """
    A masked affine autoregressive funnel layer.

    The AffineMaskedAutoregressiveInferenceFunnel is an autoregressive funnel,
    i.e., dimensionality reducing transformation, that uses an affine
    transformation from data to latent space using a masking mechanism as in
    MaskedAutoegressive.

    Examples:

        >>> import distrax
        >>> from surjectors import AffineMaskedAutoregressiveInferenceFunnel
        >>> from surjectors.nn import MADE, make_mlp
        >>> from surjectors.util import unstack
        >>>
        >>> def decoder_fn(n_dim):
        >>>     def _fn(z):
        >>>         params = make_mlp([4, 4, n_dim * 2])(z)
        >>>         mu, log_scale = jnp.split(params, 2, -1)
        >>>         return distrax.Independent(distrax.Normal(mu, jnp.exp(log_scale)))
        >>>     return _fn
        >>>
        >>> layer = AffineMaskedAutoregressiveInferenceFunnel(
        >>>     n_keep=10,
        >>>     decoder=decoder_fn(10),
        >>>     conditioner=MADE(10, [8, 8], 2),
        >>> )
    """

    def __init__(self, n_keep: int, decoder: Callable, conditioner: Callable):
        """
        Constructs a AffineMaskedAutoregressiveInferenceFunnel layer.

        Args:
            n_keep: number of dimensions to keep
            decoder: a callable that returns a conditional probabiltiy
                distribution when called
            conditioner: a MADE neural network
        """

        super().__init__(
            n_keep, decoder, conditioner, None, "inference_surjector"
        )

    def _inner_bijector(self):
        def _bijector_fn(params: Array):
            shift, log_scale = unstack(params, axis=-1)
            return distrax.ScalarAffine(shift, jnp.exp(log_scale))

        return MaskedAutoregressive(self._conditioner, _bijector_fn)

    def inverse_and_likelihood_contribution(self, y, x=None, **kwargs):
        y_plus, y_minus = y[..., : self.n_keep], y[..., self.n_keep :]

        y_cond = y_minus
        if x is not None:
            y_cond = jnp.concatenate([y_cond, x], axis=-1)
        z, jac_det = self._inner_bijector().inverse_and_log_det(y_plus, y_cond)

        z_condition = z
        if x is not None:
            z_condition = jnp.concatenate([z, x], axis=-1)
        lc = self.decoder(z_condition).log_prob(y_minus)

        return z, lc + jac_det

    def forward_and_likelihood_contribution(self, z, x=None, **kwargs):
        z_condition = z
        if x is not None:
            z_condition = jnp.concatenate([z, x], axis=-1)
        y_minus, jac_det = self.decoder(z_condition).sample_and_log_prob(
            seed=hk.next_rng_key()
        )

        y_cond = y_minus
        if x is not None:
            y_cond = jnp.concatenate([y_cond, x], axis=-1)
        # TODO(simon): need to sort the indexes correctly (?)
        # TODO(simon): remote the conditioning here?
        y_plus, lc = self._inner_bijector().forward_and_log_det(z, y_cond)

        y = jnp.concatenate([y_plus, y_minus])
        return y, lc + jac_det

    def forward(self, z, x=None):
        y, _ = self.forward_and_likelihood_contribution(z, x)
        return y