import distrax
import haiku as hk
from chex import Array
from jax import numpy as jnp

from surjectors._src.bijectors.masked_coupling import MaskedCoupling
from surjectors._src.surjectors.surjector import Surjector


class AffineMaskedCouplingGenerativeFunnel(Surjector):
    """A Generative funnel layer using masked affine coupling."""

    def __init__(self, n_keep, encoder, conditioner):
        """Construct a AffineMaskedCouplingGenerativeFunnel layer.

        Args:
            n_keep: number of dimensions to keep
            encoder: callable
            conditioner: callable
        """
        self.n_keep = n_keep
        self.encoder = encoder
        self.conditioner = conditioner

    def _mask(self, array):
        mask = jnp.arange(array.shape[-1]) >= self.n_keep
        mask = mask.astype(jnp.bool_)
        return mask

    def _inner_bijector(self, mask):
        def _bijector_fn(params: Array):
            shift, log_scale = jnp.split(params, 2, axis=-1)
            return distrax.ScalarAffine(shift, jnp.exp(log_scale))

        return MaskedCoupling(mask, self.conditioner, _bijector_fn)

    def _inverse_and_likelihood_contribution(self, y, x=None, **kwargs):
        y_condition = y
        # TODO(simon) : fixme
        if x is not None:
            y_condition = jnp.concatenate([y, x], axis=-1)
        z_minus, lc = self.encoder(y_condition).sample_and_log_prob(
            seed=hk.next_rng_key()
        )
        arr = jnp.concatenate([y, z_minus], axis=-1)
        # TODO(simon): remote the conditioning here?
        z, jac_det = self._inner_bijector(self._mask(arr)).inverse_and_log_det(
            arr
        )
        return z, -lc + jac_det

    def _forward_and_likelihood_contribution(self, z, x=None, **kwargs):
        # TODO(simon): remote the conditioning here?
        faux, jac_det = self._inner_bijector(self._mask(z)).inverse_and_log_det(
            z
        )
        y = faux[..., : self.n_keep]
        y_condition = y
        if x is not None:
            y_condition = jnp.concatenate([y_condition, x], axis=-1)
        lc = self.encoder(y_condition).log_prob(faux[..., self.n_keep :])
        return y, -lc + jac_det
