from jax import numpy as jnp

from surjectors.funnel import Funnel


class Slice(Funnel):
    def __init__(self, n_keep, kind="inference_surjection"):
        # TODO: implement decoder and encoder
        super().__init__(kind, decoder, encoder, n_keep)