from abc import ABC, abstractmethod

from jax import Array

from surjectors._src._transform import Transform


# pylint: disable=too-many-arguments
class Surjector(Transform, ABC):
    """A surjective transformation."""

    def inverse_and_likelihood_contribution(
        self, y: Array, x: Array = None, **kwargs
    ):
        """Compute the inverse transformation and its likelihood contribution.

        Args:
            y: event for which the inverse and likelihood contribution is
                computed
            x: event to condition on
            kwargs: additional keyword arguments

        Returns:
            tuple of two arrays of floats. The first one is the inverse
            transformation, the second one its likelihood contribution
        """
        return self._inverse_and_likelihood_contribution(y, x=x, **kwargs)

    @abstractmethod
    def _inverse_and_likelihood_contribution(self, y, x=None, **kwargs):
        pass

    def forward_and_likelihood_contribution(
        self, z: Array, x: Array = None, **kwargs
    ):
        """Compute the forward transformation and its likelihood contribution.

        Args:
            z: event for which the forward transform and likelihood contribution
                is computed
            x: event to condition on
            kwargs: additional keyword arguments

        Returns:
            tuple of two arrays of floats. The first one is the forward
            transformation, the second one its likelihood contribution
        """
        return self._forward_and_likelihood_contribution(z, x=x, **kwargs)

    @abstractmethod
    def _forward_and_likelihood_contribution(self, z, x=None, **kwargs):
        pass

    def forward(self, z: Array, x: Array = None, **kwargs):
        """Computes the forward transformation.

        Args:
            z: event for which the forward transform is computed
            x: event to condition on
            kwargs: additional keyword arguments

        Returns:
            result of the forward transformation
        """
        y, _ = self.forward_and_likelihood_contribution(z, x=x, **kwargs)
        return y

    def inverse(self, y: Array, x: Array = None, **kwargs):
        """Compute the inverse transformation.

        Args:
            y: event for which the inverse transform is computed
            x: event to condition on
            kwargs: additional keyword arguments

        Returns:
            result of the inverse transformation
        """
        z, _ = self.inverse_and_likelihood_contribution(y, x=x, **kwargs)
        return z
