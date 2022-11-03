from abc import abstractmethod

from jax import numpy as jnp
from surjectors.surjectors.transform import Transform


_valid_kinds = ["inference_surjector", "generative_surjector", "bijector"]


class Surjector(Transform):
    """
    Surjector base class
    """
    def __init__(self, n_keep, decoder, encoder, kind, dtype=jnp.float32):
        if kind not in _valid_kinds:
            raise ValueError(
                "'kind' argument needs to be either of: " "/".join(_valid_kinds)
            )
        if kind == _valid_kinds[1] and encoder is None:
            raise ValueError(
                "please provide an encoder if you use a generative surjection"
            )

        self._dtype = dtype
        self._kind = kind
        self._decoder = decoder
        self._encoder = encoder
        self._n_keep = n_keep

    @abstractmethod
    def inverse_and_likelihood_contribution(self, y):
        pass

    @abstractmethod
    def forward_and_likelihood_contribution(self, z):
        pass

    @property
    def n_keep(self):
        return self._n_keep

    @property
    def decoder(self):
        return self._decoder

    @property
    def encoder(self):
        return self._encoder

    @property
    def dtype(self):
        return self._dtype