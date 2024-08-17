from jax import numpy as jnp

from surjectors._src.bijectors.bijector import Bijector


# pylint: disable=arguments-renamed
class Permutation(Bijector):
    """Permute the dimensions of a vector.

    Args:
        permutation: a vector of integer indexes representing the order of
            the elements
        event_ndims_in: number of input event dimensions

    Examples:
        >>> from surjectors import Permutation
        >>> from jax import numpy as jnp
        >>>
        >>> order = jnp.arange(10)
        >>> perm = Permutation(order, 1)
    """

    def __init__(self, permutation, event_ndims_in: int):
        self.permutation = permutation
        self.event_ndims_in = event_ndims_in

    def _forward_and_likelihood_contribution(self, z, **kwargs):
        return z[..., self.permutation], jnp.full(jnp.shape(z)[:-1], 0.0)

    def _inverse_and_likelihood_contribution(self, y, **kwargs):
        size = self.permutation.size
        permutation_inv = (
            jnp.zeros(size, dtype=jnp.result_type(int))
            .at[self.permutation]
            .set(jnp.arange(size))
        )
        return y[..., permutation_inv], jnp.full(jnp.shape(y)[:-1], 0.0)
