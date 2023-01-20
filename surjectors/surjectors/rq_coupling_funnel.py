import distrax
from chex import Array
from distrax import MaskedCoupling
from jax import numpy as jnp

from surjectors.surjectors.surjector import Surjector


# pylint: disable=too-many-arguments
class NSFCouplingFunnel(Surjector):
    """
    Neural spline flow coupling funnel


    """

    def __init__(
        self,
        n_keep,
        decoder,
        conditioner,
        range_min,
        range_max,
    ):
        super().__init__(n_keep, decoder, None, "inference_surjection")
        self._conditioner = conditioner
        self._range_min = range_min
        self._range_max = range_max

    def _mask(self, array):
        mask = jnp.arange(array.shape[-1]) >= self.n_keep
        mask = mask.astype(jnp.bool_)
        return mask

    def _inner_bijector(self, mask):
        def _bijector_fn(params: Array):
            return distrax.RationalQuadraticSpline(
                params, range_min=self._range_min, range_max=self._range_max
            )

        return MaskedCoupling(mask, self._conditioner, _bijector_fn)

    def inverse_and_likelihood_contribution(self, y, x: Array = None):
        # TODO(simon: this needs to be implemted
        mask = self._mask(y)
        faux, jac_det = self._inner_bijector(mask).inverse_and_log_det(y)
        z = faux[:, : self.n_keep]
        lp = self.decoder.log_prob(faux[:, self.n_keep :], context=z)
        return z, lp + jac_det

    def forward_and_likelihood_contribution(self, z, x: Array = None):
        # TODO(simon: this needs to be implemted
        raise NotImplementedError()
