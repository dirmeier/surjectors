import distrax
from chex import Array

# pylint: disable=too-many-arguments, arguments-renamed
from surjectors.surjectors.masked_coupling_inference_funnel import (
    MaskedCouplingInferenceFunnel,
)


class RationalQuadraticSplineMaskedCouplingInferenceFunnel(
    MaskedCouplingInferenceFunnel
):
    """
    Inference funnel layer using masked spline coupling
    """

    def __init__(self, n_keep, decoder, conditioner, range_min, range_max):
        self.range_min = range_min
        self.range_max = range_max

        def _bijector_fn(params: Array):
            return distrax.RationalQuadraticSpline(
                params, self.range_min, self.range_max
            )

        super().__init__(n_keep, decoder, conditioner, _bijector_fn)
