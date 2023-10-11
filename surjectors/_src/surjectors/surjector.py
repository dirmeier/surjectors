from abc import ABC, abstractmethod

from chex import Array
from jax import numpy as jnp

from surjectors.surjectors._transform import Transform

_valid_kinds = [
    "inference_surjector",
    "generative_surjector",
    "bijector",
    "surjector",
]


# pylint: disable=too-many-arguments
class Surjector(Transform, ABC):
    """
    Surjector base class
    """

    def __init__(self, n_keep, decoder, encoder, kind, dtype=jnp.float32):
        if kind not in _valid_kinds:
            raise ValueError(
                "'kind' argument needs to be either of: "
                + "/".join(_valid_kinds)
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
    def inverse_and_likelihood_contribution(self, y: Array, **kwargs):
        pass

    @abstractmethod
    def forward_and_likelihood_contribution(self, z: Array, **kwargs):
        pass

    @property
    def n_keep(self):
        """
        Returns the Number of elements of the inverse transformation to keep
        """

        return self._n_keep

    @property
    def dtype(self):
        """Returns the dtype of the transformation"""

        return self._dtype

    @property
    def decoder(self):
        """Returns the decoder network"""

        return self._decoder

    @property
    def encoder(self):
        """Returns the encoder network"""

        return self._encoder
