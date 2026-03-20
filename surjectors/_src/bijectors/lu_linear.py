import haiku as hk
import jax.nn
import numpy as np
from jax import numpy as jnp

from surjectors._src.bijectors.bijector import Bijector


# pylint: disable=arguments-differ,too-many-instance-attributes
class LULinear(Bijector, hk.Module):
  """An bijection based on the LU composition.

  Args:
    n_dimension: number of dimensions to keep
    with_bias: use a bias term or not
    dtype: parameter dtype

  References:
    .. [1] Oliva, Junier, et al. "Transformation Autoregressive Networks".
       Proceedings of the 35th International Conference on
      Machine Learning, 2018.

  Examples:
    >>> import haiku as hk
    >>> from jax import random as jr
    >>> from tensorflow_probability.substrates.jax import distributions as tfd
    >>> from surjectors import LULinear, TransformedDistribution

    >>> @hk.without_apply_rng
    ... @hk.transform
    ... def fn(inputs):
    ...   base_distribution = tfd.Independent(
    ...     tfd.Normal(jnp.zeros(5), jnp.ones(5)),
    ...     reinterpreted_batch_ndims=1,
    ...   )
    ...   td = TransformedDistribution(base_distribution, LULinear(5))
    ...   return td.log_prob(inputs)
    >>>
    >>> data = jr.normal(jr.PRNGKey(1), shape=(10, 5))
    >>> params = fn.init(jr.key(0), data)
    >>> lps = fn.apply(params, data)
  """

  def __init__(self, n_dimension, with_bias=False, dtype=jnp.float32):
    super().__init__()
    if with_bias:
      raise NotImplementedError()

    self.n_dimension = n_dimension
    self.with_bias = with_bias
    self.dtype = dtype
    n_triangular_entries = ((n_dimension - 1) * n_dimension) // 2

    self._lower_indices = np.tril_indices(n_dimension, k=-1)
    self._upper_indices = np.triu_indices(n_dimension, k=1)
    self._diag_indices = np.diag_indices(n_dimension)

    self._lower_entries = hk.get_parameter(
      "lower_entries", [n_triangular_entries], dtype=dtype, init=jnp.zeros
    )
    self._upper_entries = hk.get_parameter(
      "upper_entries", [n_triangular_entries], dtype=dtype, init=jnp.zeros
    )
    self._unconstrained_upper_diag_entries = hk.get_parameter(
      "diag_entries", [n_dimension], dtype=dtype, init=jnp.ones
    )

  def _to_lower_and_upper_matrices(self):
    L = jnp.zeros((self.n_dimension, self.n_dimension), dtype=self.dtype)
    L = L.at[self._lower_indices].set(self._lower_entries)
    L = L.at[self._diag_indices].set(1.0)

    U = jnp.zeros((self.n_dimension, self.n_dimension), dtype=self.dtype)
    U = U.at[self._upper_indices].set(self._upper_entries)
    U = U.at[self._diag_indices].set(self._upper_diag)

    return L, U

  @property
  def _upper_diag(self):
    return jax.nn.softplus(self._unconstrained_upper_diag_entries) + 1e-4

  def _inverse_likelihood_contribution(self):
    return jnp.sum(jnp.log(self._upper_diag))

  def _inverse_and_likelihood_contribution(self, y, x=None, **kwargs):
    L, U = self._to_lower_and_upper_matrices()
    z = jnp.dot(jnp.dot(y, U), L)
    lc = self._inverse_likelihood_contribution() * jnp.ones(z.shape[0])
    return z, lc

  def _forward_and_likelihood_contribution(self, z, x=None, **kwargs):
    raise NotImplementedError()
