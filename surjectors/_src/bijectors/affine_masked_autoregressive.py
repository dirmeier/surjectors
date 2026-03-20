import distrax
from jax import numpy as jnp

from surjectors._src.bijectors.masked_autoregressive import MaskedAutoregressive
from surjectors._src.conditioners.nn.made import MADE
from surjectors.util import unstack


# pylint: disable=too-many-arguments,arguments-renamed
class AffineMaskedAutoregressive(MaskedAutoregressive):
  """An affine masked autoregressive layer.

  Args:
      conditioner: a MADE network
      event_ndims: the number of array dimensions the bijector operates on
      inner_event_ndims: tthe number of array dimensions the bijector
          operates on

  References:
      .. [1] Papamakarios, George, et al. "Masked Autoregressive Flow for
          Density Estimation". Advances in Neural Information Processing
          Systems, 2017.

  Examples:
      >>> import haiku as hk
      >>> from jax import random as jr
      >>> from tensorflow_probability.substrates.jax import distributions as tfd
      >>> from surjectors import AffineMaskedAutoregressive, TransformedDistribution

      >>> @hk.without_apply_rng
      ... @hk.transform
      ... def fn(inputs):
      ...   base_distribution = tfd.Independent(
      ...     tfd.Normal(jnp.zeros(10), jnp.ones(10)),
      ...     reinterpreted_batch_ndims=1,
      ...   )
      ...   td = TransformedDistribution(
      ...     base_distribution,
      ...     AffineMaskedAutoregressive(MADE(10, (64, 64), 2))
      ...  )
      ...   return td.log_prob(inputs)

      >>> data = jr.normal(jr.PRNGKey(1), shape=(10, 10))
      >>> params = fn.init(jr.key(0), data)
      >>> lps = fn.apply(params, data)
  """

  def __init__(
    self,
    conditioner: MADE,
    event_ndims: int = 1,
    inner_event_ndims: int = 0,
  ):
    def bijector_fn(params):
      means, log_scales = unstack(params, -1)
      return distrax.ScalarAffine(means, jnp.exp(log_scales))

    super().__init__(conditioner, bijector_fn, event_ndims, inner_event_ndims)
