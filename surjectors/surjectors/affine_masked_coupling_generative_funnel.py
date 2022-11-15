import distrax
from chex import Array
from jax import numpy as jnp
import haiku as hk
from surjectors.bijectors.masked_coupling import MaskedCoupling
from surjectors.surjectors.funnel import Funnel


class AffineMaskedCouplingGenerativeFunnel(Funnel):
    def __init__(self, n_keep, encoder, conditioner):
        super().__init__(n_keep, None, conditioner, encoder, "generative_surjector")

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
        y_condition = y
        # TODO
        if x is not None:
            y_condition = jnp.concatenate([y, x], axis=-1)
        z_minus, lc = self.encoder(y_condition).sample_and_log_prob(seed=hk.next_rng_key())
        input = jnp.concatenate([y, z_minus], axis=-1)
        # TODO: remote the conditioning here?
        z, jac_det = self._inner_bijector(self._mask(input)).inverse_and_log_det(input)
        return z, -lc + jac_det

    def forward_and_likelihood_contribution(self, z, x=None):
        # TODO: remote the conditioning here?
        faux, jac_det = self._inner_bijector(self._mask(z)).inverse_and_log_det(z)
        y = faux[..., :self.n_keep]
        y_condition = y
        if x is not None:
            y_condition = jnp.concatenate([y_condition, x], axis=-1)
        lc = self.encoder(y_condition).log_prob(faux[..., self.n_keep:])
        return y, -lc + jac_det

    def forward(self, z, x=None):
        y, _ = self.forward_and_likelihood_contribution(z, x)
        return y
