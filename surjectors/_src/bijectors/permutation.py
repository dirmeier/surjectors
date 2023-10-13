import distrax
from jax import numpy as jnp


# pylint: disable=arguments-renamed
class Permutation(distrax.Bijector):
    """Permute the dimensions of a vector.

    Examples:
        >>> from surjectors import Permutation
        >>> from jax import numpy as jnp
        >>>
        >>> order = jnp.arange(10)
        >>> perm = Permutation(order, 1)
    """

    def __init__(self, permutation, event_ndims_in: int):
        """Construct a permutation layer.

        Args:
            permutation: a vector of integer indexes representing the order of
                the elements
            event_ndims_in: number of input event dimensions
        """
        super().__init__(event_ndims_in)
        self.permutation = permutation

    def forward_and_log_det(self, z):
        """Compute the forward transformation and its Jacobian determinant.

        Args:
           z: event for which the forward transform and likelihood contribution
               is computed

        Returns:
           tuple of two arrays of floats. The first one is the forward
           transformation, the second one its likelihood contribution
        """
        return z[..., self.permutation], jnp.full(jnp.shape(z)[:-1], 0.0)

    def inverse_and_log_det(self, y):
        """Compute the inverse transformation and its Jacobian determinant.

        Args:
            y: event for which the inverse and likelihood contribution is
                computed

        Returns:
            tuple of two arrays of floats. The first one is the inverse
            transformation, the second one its likelihood contribution
        """
        size = self.permutation.size
        permutation_inv = (
            jnp.zeros(size, dtype=jnp.result_type(int))
            .at[self.permutation]
            .set(jnp.arange(size))
        )
        return y[..., permutation_inv], jnp.full(jnp.shape(y)[:-1], 0.0)
