from abc import ABC

from chex import Array

from surjectors._src.surjectors.surjector import Surjector


# pylint: disable=too-many-arguments
class Bijector(Surjector, ABC):
    """Bijector base class."""

    def inverse_and_log_det(self, y: Array, x: Array = None, **kwargs):
        """Compute the inverse transformation and its Jacobian determinant.

        Args:
            y: event for which the inverse and likelihood contribution is
                computed
            x: event to condition on
            kwargs: additional keyword arguments

        Returns:
            tuple of two arrays of floats. The first one is the inverse
            transformation, the second one its likelihood contribution
        """
        return self.inverse_and_likelihood_contribution(y, x=x, **kwargs)

    def forward_and_log_det(self, z: Array, x: Array = None, **kwargs):
        """Compute the forward transformation and its Jacobian determinant.

        Args:
            z: event for which the forward transform and likelihood contribution
                is computed
            x: event to condition on
            kwargs: additional keyword arguments

        Returns:
            tuple of two arrays of floats. The first one is the forward
            transformation, the second one its likelihood contribution
        """
        return self.forward_and_likelihood_contribution(z, x=x, **kwargs)
