import distrax
from chex import Array

# pylint: disable=too-many-arguments, arguments-renamed
from surjectors._src.surjectors.masked_coupling_inference_funnel import (
  MaskedCouplingInferenceFunnel,
)


# ruff: noqa: PLR0913
class RationalQuadraticSplineMaskedCouplingInferenceFunnel(
  MaskedCouplingInferenceFunnel
):
  """A masked coupling inference funnel that uses a rational quatratic spline.

  Args:
      n_keep: number of dimensions to keep
      decoder: a callable that returns a conditional probabiltiy
          distribution when called
      conditioner: a conditioning neural network
      range_min: minimum range of the spline
      range_max: maximum range of the spline

  Examples:
      >>> import haiku as hk
      >>> from jax import numpy as jnp
      >>> from jax import random as jr
      >>> from tensorflow_probability.substrates.jax import distributions as tfd
      >>> from surjectors import TransformedDistribution
      >>> from surjectors import RationalQuadraticSplineMaskedCouplingInferenceFunnel
      >>> from surjectors.nn import make_mlp
      >>>
      >>> def decoder_fn(n_dim):
      ...     def _fn(z):
      ...         params = make_mlp((64, 64, n_dim * 2))(z)
      ...         mu, log_scale = jnp.split(params, 2, -1)
      ...         return tfd.Independent(
      ...             tfd.Normal(mu, jnp.exp(log_scale))
      ...         )
      ...     return _fn
      >>>
      >>> @hk.without_apply_rng
      ... @hk.transform
      ... def fn(inputs, num_params=3):
      ...   base_distribution = tfd.Independent(
      ...     tfd.Normal(jnp.zeros(4), jnp.ones(4)),
      ...     reinterpreted_batch_ndims=1,
      ...   )
      ...   td = TransformedDistribution(
      ...     base_distribution,
      ...     RationalQuadraticSplineMaskedCouplingInferenceFunnel(
      ...       n_keep=4,
      ...       decoder=decoder_fn(10 - 4),
      ...       conditioner=hk.Sequential([
      ...         make_mlp((64, 64, 10 * (3 * num_params + 1))),
      ...         hk.Reshape((10, 3 * num_params + 1), preserve_dims=-1)
      ...       ]),
      ...       range_min=-2, range_max=2
      ...     )
      ...   )
      ...   return td.log_prob(inputs)
      >>>
      >>> data = jr.normal(jr.PRNGKey(1), shape=(10, 10))
      >>> params = fn.init(jr.key(0), data)
      >>> lps = fn.apply(params, data)

  References:
      .. [1] Klein, Samuel, et al. "Funnels: Exact maximum likelihood
          with dimensionality reduction". Workshop on Bayesian Deep Learning,
          Advances in Neural Information Processing Systems, 2021.
      .. [2] Durkan, Conor, et al. "Neural Spline Flows".
          Advances in Neural Information Processing Systems, 2019.
      .. [3] Dinh, Laurent, et al. "Density estimation using RealNVP".
          International Conference on Learning Representations, 2017.
  """

  def __init__(self, n_keep, decoder, conditioner, range_min, range_max):
    self.range_min = range_min
    self.range_max = range_max

    def _bijector_fn(params: Array):
      return distrax.RationalQuadraticSpline(
        params, self.range_min, self.range_max
      )

    super().__init__(n_keep, decoder, conditioner, _bijector_fn)
