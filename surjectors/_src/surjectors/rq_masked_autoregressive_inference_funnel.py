import warnings

import distrax
from chex import Array

from surjectors.bijectors.masked_autoregressive import MaskedAutoregressive
from surjectors.surjectors.affine_masked_autoregressive_inference_funnel import (  # noqa: E501
    AffineMaskedAutoregressiveInferenceFunnel,
)


# pylint: disable=too-many-arguments, arguments-renamed
class RationalQuadraticSplineMaskedAutoregressiveInferenceFunnel(
    AffineMaskedAutoregressiveInferenceFunnel
):
    """
    Inference funnel layer using masked rational quadratic autoregression
    """

    def __init__(self, n_keep, decoder, conditioner, range_min, range_max):
        warnings.warn("class has not been tested. use at own risk")
        super().__init__(n_keep, decoder, conditioner)
        self.range_min = range_min
        self.range_max = range_max

    def _inner_bijector(self):
        def _bijector_fn(params: Array):
            return distrax.RationalQuadraticSpline(
                params, self.range_min, self.range_max
            )

        return MaskedAutoregressive(self._conditioner, _bijector_fn)
