from abc import ABC, abstractmethod

from chex import Array
from jax import numpy as jnp

from surjectors.surjectors.surjector import Surjector


# pylint: disable=too-many-arguments
class _Bijector(Surjector):
    """
    Bijector base class
    """

    def __init__(self, conditioner, bijector_fn, dtype=jnp.float32):
        super().__init__(None, conditioner, bijector_fn, "bijector", dtype)

    @property
    def bijector_fn(self):
        return self._encoder

    def _inner_bijector(self, params):
        return self.bijector_fn(params)

    @property
    def conditioner(self):
        return self._decoder

    @property
    def decoder(self):
        raise NotImplementedError("")

    @property
    def encoder(self):
        raise NotImplementedError("")
