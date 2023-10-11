import distrax
from jax import numpy as jnp


# pylint: disable=arguments-renamed
class Permutation(distrax.Bijector):
    """
    Permute the dimensions of a vector
    """

    def __init__(self, permutation, event_ndims_in: int):
        super().__init__(event_ndims_in)
        self.permutation = permutation

    def forward_and_log_det(self, z):
        return z[..., self.permutation], jnp.full(jnp.shape(z)[:-1], 0.0)

    def inverse_and_log_det(self, y):
        size = self.permutation.size
        permutation_inv = (
            jnp.zeros(size, dtype=jnp.result_type(int))
            .at[self.permutation]
            .set(jnp.arange(size))
        )
        return y[..., permutation_inv], jnp.full(jnp.shape(y)[:-1], 0.0)
