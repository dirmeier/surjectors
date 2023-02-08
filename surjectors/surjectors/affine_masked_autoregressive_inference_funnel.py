import distrax
import haiku as hk
from chex import Array
from jax import numpy as jnp

from surjectors.bijectors.masked_autoregressive import MaskedAutoregressive
from surjectors.surjectors.funnel import Funnel
from surjectors.util import unstack


class AffineMaskedAutoregressiveInferenceFunnel(Funnel):
    """
    Inference funnel layer using masked affine autoregression
    """

    def __init__(self, n_keep, decoder, conditioner):
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
