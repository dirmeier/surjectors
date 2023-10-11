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
        """Returns the bijector function"""
        return self._encoder

    def _inner_bijector(self, params):
        return self.bijector_fn(params)

    @property
    def conditioner(self):
        """Returns the conditioner network"""
        return self._decoder

    @property
    def decoder(self):
        raise NotImplementedError("")

    @property
    def encoder(self):
        raise NotImplementedError("")

    def forward_and_log_det(self, z, x: Array = None):
        """
        Compute the forward transformation and its likelihood contribution

        Parameters
        ----------
        z: jnp.ndarray
            event for which the forward transform and likelihood contribution
            is computed
        x: Optional[jnp.ndarray]
            event to condition on

        Returns
        -------
        Tuple[jnp.ndarray, jnp.ndarray]
            tuple of two arrays of floats. The first one is the forward
            transformation, the second one its likelihood contribution
        """

    def inverse_and_log_det(self, y, x: Array = None):
        """
        Compute the inverse transformation and its likelihood contribution

        Parameters
        ----------
        y: jnp.ndarray
            event for which the inverse and likelihood contribution is computed
        x: Optional[jnp.ndarray]
            event to condition on

        Returns
        -------
        Tuple[jnp.ndarray, jnp.ndarray]
            tuple of two arrays of floats. The first one is the inverse
            transformation, the second one its likelihood contribution
        """
