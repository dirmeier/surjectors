import haiku as hk
import jax.nn
import numpy as np
from jax import numpy as jnp

from surjectors._src.surjectors.surjector import Surjector


# pylint: disable=arguments-differ,too-many-instance-attributes
class LULinear(Surjector):
    """An inference funnel based on the LU composition.

    References:
        .. [1] Klein, Samuel, et al. "Funnels: Exact maximum likelihood
            with dimensionality reduction". Workshop on Bayesian Deep Learning,
            Advances in Neural Information Processing Systems, 2021.

    Examples:
        >>> from surjectors import LULinear
        >>> layer = LULinear(10)
    """

    def __init__(self, n_keep, with_bias=False, dtype=jnp.float32):
        """Constructs a LULinear layer.

        Args:
            n_keep: number of dimensions to keep
            with_bias: use a bias term or not
            dtype: parameter dtype
        """
        if with_bias:
            raise NotImplementedError()

        self.n_keep = n_keep
        self.with_bias = with_bias
        self.dtype = dtype
        n_triangular_entries = ((n_keep - 1) * n_keep) // 2

        self._lower_indices = np.tril_indices(n_keep, k=-1)
        self._upper_indices = np.triu_indices(n_keep, k=1)
        self._diag_indices = np.diag_indices(n_keep)

        self._lower_entries = hk.get_parameter(
            "lower_entries", [n_triangular_entries], dtype=dtype, init=jnp.zeros
        )
        self._upper_entries = hk.get_parameter(
            "upper_entries", [n_triangular_entries], dtype=dtype, init=jnp.zeros
        )
        self._unconstrained_upper_diag_entries = hk.get_parameter(
            "diag_entries", [n_keep], dtype=dtype, init=jnp.ones
        )

    def _to_lower_and_upper_matrices(self):
        L = jnp.zeros((self.n_keep, self.n_keep), dtype=self.dtype)
        L = L.at[self._lower_indices].set(self._lower_entries)
        L = L.at[self._diag_indices].set(1.0)

        U = jnp.zeros((self.n_keep, self.n_keep), dtype=self.dtype)
        U = U.at[self._upper_indices].set(self._upper_entries)
        U = U.at[self._diag_indices].set(self._upper_diag)

        return L, U

    @property
    def _upper_diag(self):
        return jax.nn.softplus(self._unconstrained_upper_diag_entries) + 1e-4

    def _inverse_likelihood_contribution(self):
        return jnp.sum(jnp.log(self._upper_entries))

    def _inverse_and_likelihood_contribution(self, y, x=None, **kwargs):
        L, U = self._to_lower_and_upper_matrices()
        z = jnp.dot(jnp.dot(y, U), L)
        lc = self._inverse_likelihood_contribution()
        return z, lc * jnp.ones_like(z)

    def _forward_and_likelihood_contribution(self, z, x=None, **kwargs):
        raise NotImplementedError()
