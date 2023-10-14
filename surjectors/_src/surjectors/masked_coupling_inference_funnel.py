from typing import Callable

import haiku as hk
from jax import numpy as jnp

from surjectors._src.bijectors.masked_coupling import MaskedCoupling
from surjectors._src.surjectors.surjector import Surjector


# pylint: disable=too-many-function-args
class MaskedCouplingInferenceFunnel(Surjector):
    """A masked coupling inference funnel.

    The MaskedCouplingInferenceFunnel is a coupling funnel,
    i.e., dimensionality reducing transformation, that uses a masking mechanism
    as in MaskedCouplingI. Its inner bijectors needs to be specified in
    comparison to ASffineMaskedCouplingInferenceFunnel and
    RationalQuadraticSplineMaskedCouplingInferenceFunnel.

    References:
        .. [1] Klein, Samuel, et al. "Funnels: Exact maximum likelihood
            with dimensionality reduction". Workshop on Bayesian Deep Learning,
            Advances in Neural Information Processing Systems, 2021.
        .. [2] Dinh, Laurent, et al. "Density estimation using RealNVP".
            International Conference on Learning Representations, 2017.

    Examples:
        >>> import distrax
        >>> from surjectors import MaskedCouplingInferenceFunnel
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
        >>> def bijector_fn(params):
        >>>     shift, log_scale = jnp.split(params, 2, -1)
        >>>     return distrax.ScalarAffine(shift, jnp.exp(log_scale))
        >>>
        >>> layer = MaskedCouplingInferenceFunnel(
        >>>     n_keep=10,
        >>>     decoder=decoder_fn(10),
        >>>     conditioner=make_mlp([4, 4, 10 * 2]),
        >>>     bijector_fn=bijector_fn
        >>> )
    """

    def __init__(
        self,
        n_keep: int,
        decoder: Callable,
        conditioner: Callable,
        bijector_fn: Callable,
    ):
        """Construct a MaskedCouplingInferenceFunnel layer.

        Args:
            n_keep: number of dimensions to keep
            decoder: a callable that returns a conditional probabiltiy
                distribution when called
            conditioner: a conditioning neural network
            bijector_fn: an inner bijector function to be used
        """
        self.n_keep = n_keep
        self.decoder = decoder
        self.conditioner = conditioner
        self.bijector_fn = bijector_fn

    def _mask(self, array):
        mask = jnp.arange(array.shape[-1]) >= self.n_keep
        mask = mask.astype(jnp.bool_)
        return mask

    def _inner_bijector(self, mask):
        return MaskedCoupling(mask, self.conditioner, self.bijector_fn)

    def _inverse_and_likelihood_contribution(self, y, x=None, **kwargs):
        # TODO(simon): remote the conditioning here?
        faux, jac_det = self._inner_bijector(self._mask(y)).inverse_and_log_det(
            y, x=x
        )
        # TODO(simon): this should be relegated to the base class:
        #  MaskedCouplingInferenceFunnel (see issue #14 on GitHub)
        z_condition = z = faux[:, : self.n_keep]
        if x is not None:
            z_condition = jnp.concatenate([z, x], axis=-1)
        lc = self.decoder(z_condition).log_prob(y[:, self.n_keep :])
        return z, lc + jac_det

    def _forward_and_likelihood_contribution(self, z, x=None, **kwargs):
        z_condition = z
        if x is not None:
            z_condition = jnp.concatenate([z, x], axis=-1)
        y_minus, jac_det = self.decoder(z_condition).sample_and_log_prob(
            seed=hk.next_rng_key()
        )
        # TODO(simon): need to sort the indexes correctly (?)
        z_tilde = jnp.concatenate([z, y_minus], axis=-1)
        # TODO(simon): remote the conditioning here?
        y, lc = self._inner_bijector(self._mask(z_tilde)).forward_and_log_det(
            z_tilde, x
        )
        return y, lc + jac_det
