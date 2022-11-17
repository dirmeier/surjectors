import distrax
from chex import Array
from jax import numpy as jnp
import haiku as hk
from surjectors.bijectors.masked_coupling import MaskedCoupling
from surjectors.surjectors.funnel import Funnel


class AffineMaskedCouplingInferenceFunnel(Funnel):
    def __init__(self, n_keep, decoder, conditioner):
        super().__init__(n_keep, decoder, conditioner, None, "inference_surjector")

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

    def inverse_and_likelihood_contribution(self, y, x=None):
        # TODO: remote the conditioning here?
        faux, jac_det = self._inner_bijector(self._mask(y)).inverse_and_log_det(y, x)
        z_condition = z = faux[:, :self.n_keep]
        if x is not None:
            z_condition = jnp.concatenate([z, x], axis=-1)
        lc = self.decoder(z_condition).log_prob(y[:, self.n_keep:])
        return z, lc + jac_det

    def forward_and_likelihood_contribution(self, z, x=None):
        z_condition = z
        if x is not None:
            z_condition = jnp.concatenate([z, x], axis=-1)
        y_minus, jac_det = self.decoder(z_condition).sample_and_log_prob(seed=hk.next_rng_key())
        # TODO need to sort the indexes correctly (?)
        z_tilde = jnp.concatenate([z, y_minus], axis=-1)
        # TODO: remote the conditioning here?
        y, lc = self._inner_bijector(self._mask(z_tilde)).forward_and_log_det(z_tilde, x)
        return y, lc + jac_det

    def forward(self, z, x=None):
        y, _ = self.forward_and_likelihood_contribution(z, x)
        return y