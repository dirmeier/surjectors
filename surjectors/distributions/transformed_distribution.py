from typing import Tuple

import jax
import jax.numpy as jnp
from distrax._src.bijectors import bijector as bjct_base
from distrax._src.distributions import distribution as dist_base
from distrax._src.utils import conversion
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

PRNGKey = dist_base.PRNGKey
Array = dist_base.Array
DistributionLike = dist_base.DistributionLike
BijectorLike = bjct_base.BijectorLike


class TransformedDistribution:
    def __init__(self, base_distribution, surjector):
        self.base_distribution = base_distribution
        self.surjector = surjector

    def log_prob(self, y: Array) -> Array:
        x, ildj_y = self.surjector.inverse_and_log_det(y)
        lp_x = self.base_distribution.log_prob(x)
        lp_y = lp_x + ildj_y
        return lp_y

    def sample(self, key: PRNGKey, sample_shape=(1,)):
        z = self.base_distribution.sample(seed=key, sample_shape=sample_shape)
        y = jax.vmap(self.surjector.inverse)(z)
        return y

    def sample_and_log_prob(self, key: PRNGKey, sample_shape=(1,)):
        z, lp_z = self.base_distribution.sample_and_log_prob(
            seed=key, sample_shape=sample_shape
        )
        y, fldj = jax.vmap(self.surjector.forward_and_log_det)(z)
        lp_y = jax.vmap(jnp.subtract)(lp_z, fldj)
        return y, lp_y
