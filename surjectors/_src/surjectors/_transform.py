from abc import ABCMeta, abstractmethod

import chex
from distrax._src.utils import jittable


class Transform(jittable.Jittable, metaclass=ABCMeta):
    """
    A transformation of a random variable
    """

    @abstractmethod
    def inverse_and_likelihood_contribution(self, y: chex.Array):
        """
        Compute the inverse transformation and its likelihood contribution

        Parameters
        ----------
        y: jnp.ndarray
            event for which the inverse and likelihood contribution is computed

        Returns
        -------
        Tuple[jnp.ndarray, jnp.ndarray]
            tuple of two arrays of floats. The first one is the inverse
            transformation, the second one its likelihood contribution
        """

    @abstractmethod
    def forward_and_likelihood_contribution(self, z: chex.Array):
        """
        Compute the forward transformation and its likelihood contribution

        Parameters
        ----------
        z: jnp.ndarray
            event for which the forward transform and likelihood contribution
            is computed

        Returns
        -------
        Tuple[jnp.ndarray, jnp.ndarray]
            tuple of two arrays of floats. The first one is the forward
            transformation, the second one its likelihood contribution
        """

    def forward(self, z: chex.Array):
        """
        Compute the forward transformation

        Parameters
        ----------
        z: jnp.ndarray
            event for which the forward transform is computed

        Returns
        -------
        jnp.ndarray
            result of the forward transformation
        """

        y, _ = self.forward_and_likelihood_contribution(z)
        return y

    def inverse(self, y: chex.Array):
        """
        Compute the inverse transformation

        Parameters
        ----------
        y: jnp.ndarray
            event for which the inverse transform is computed

        Returns
        -------
        jnp.ndarray
            result of the inverse transformation
        """

        z, _ = self.inverse_and_likelihood_contribution(y)
        return z
