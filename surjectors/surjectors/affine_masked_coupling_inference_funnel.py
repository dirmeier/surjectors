import distrax
from chex import Array
from jax import numpy as jnp

from surjectors.surjectors.masked_coupling_inference_funnel import (
    MaskedCouplingInferenceFunnel,
)


class AffineMaskedCouplingInferenceFunnel(MaskedCouplingInferenceFunnel):
    """
    Inference funnel layer using masked affine coupling
    """

    def __init__(self, n_keep, decoder, conditioner):
        def _bijector_fn(params: Array):
            shift, log_scale = jnp.split(params, 2, axis=-1)
            return distrax.ScalarAffine(shift, jnp.exp(log_scale))

        super().__init__(n_keep, decoder, conditioner, _bijector_fn)
