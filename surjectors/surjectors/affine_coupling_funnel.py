import chex
import distrax
import numpy as np
from chex import Array
from distrax import ScalarAffine, MaskedCoupling
from jax import numpy as jnp

from surjectors.surjectors.surjector import Surjector


class AffineCouplingFunnel(Surjector):
    def __init__(self, n_keep, decoder, conditioner, kind="inference_surjection"):
        super().__init__(n_keep, decoder, None, kind)
        self._conditioner = conditioner

    def _mask(self, array):
        mask = jnp.arange(array.shape[-1]) >= self.n_keep
        mask = mask.astype(jnp.bool_)
        return mask

    def _inner_bijector(self, mask):
        def _bijector_fn(params: Array):
            shift, log_scale = jnp.split(params, 2, axis=-1)
            return distrax.ScalarAffine(shift, jnp.exp(log_scale))

        return MaskedCoupling(
            mask, self._conditioner, _bijector_fn
        )

    def inverse_and_likelihood_contribution(self, y):
        mask = self._mask(y)
        faux, jac_det = self._inner_bijector(mask).inverse_and_log_det(y)
        z = faux[:, :self.n_keep]
        lp = self.decoder.log_prob(faux[:, self.n_keep:], context=z)
        return z, lp + jac_det

    def forward_and_likelihood_contribution(self, z):
        raise NotImplementedError()

