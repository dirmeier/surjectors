from typing import Callable

import haiku as hk
from jax import numpy as jnp

from surjectors._src.bijectors.masked_autoregressive import MaskedAutoregressive
from surjectors._src.conditioners.nn.made import MADE
from surjectors._src.surjectors.surjector import Surjector


class MaskedAutoregressiveInferenceFunnel(Surjector):
    """A masked autoregressive funnel layer.

    The MaskedAutoregressiveInferenceFunnel is an autoregressive funnel,
    i.e., dimensionality reducing transformation, that uses a masking mechanism
    as in MaskedAutoegressive. Its inner bijectors needs to be specified in
    comparison to AffineMaskedAutoregressiveInferenceFunnel and
    RationalQuadraticSplineMaskedAutoregressiveInferenceFunnel.

    References:
        .. [1] Klein, Samuel, et al. "Funnels: Exact maximum likelihood
            with dimensionality reduction". Workshop on Bayesian Deep Learning,
            Advances in Neural Information Processing Systems, 2021.
        .. [2] Papamakarios, George, et al. "Masked Autoregressive Flow for
            Density Estimation". Advances in Neural Information Processing
            Systems, 2017.

    Examples:
        >>> import distrax
        >>> from surjectors import MaskedAutoregressiveInferenceFunnel
        >>> from surjectors.nn import MADE, make_mlp
        >>> from surjectors.util import unstack
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
        >>> def bijector_fn(params: Array):
        >>>     shift, log_scale = unstack(params, axis=-1)
        >>>     return distrax.ScalarAffine(shift, jnp.exp(log_scale))
        >>>
        >>> layer = MaskedAutoregressiveInferenceFunnel(
        >>>     n_keep=10,
        >>>     decoder=decoder_fn(10),
        >>>     conditioner=MADE(10, [8, 8], 2),
        >>>     bijector_fn=bijector_fn
        >>> )
    """

    def __init__(
        self,
        n_keep: int,
        decoder: Callable,
        conditioner: MADE,
        bijector_fn: Callable,
    ):
        """Constructs a MaskedAutoregressiveInferenceFunnel layer.

        Args:
            n_keep: number of dimensions to keep
            decoder: a callable that returns a conditional probabiltiy
                distribution when called
            conditioner: a MADE neural network
            bijector_fn: an inner bijector function to be used
        """
        self.n_keep = n_keep
        self.decoder = decoder
        self.conditioner = conditioner
        self.bijector_fn = bijector_fn

    def _inner_bijector(self):
        return MaskedAutoregressive(self.conditioner, self.bijector_fn)

    def _inverse_and_likelihood_contribution(self, y, x=None, **kwargs):
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

    def _forward_and_likelihood_contribution(self, z, x=None, **kwargs):
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
