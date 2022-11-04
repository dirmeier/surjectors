import chex
import jax
import jax.numpy as jnp
from chex import PRNGKey
from distrax import Distribution
Array = chex.Array
from surjectors.surjectors.surjector import Surjector


class TransformedDistribution:
    def __init__(self, base_distribution: Distribution, surjector: Surjector):
        self.base_distribution = base_distribution
        self.surjector = surjector

    def log_prob(self, y: Array) -> jnp.ndarray:
        x, lc = self.surjector.inverse_and_likelihood_contribution(y)
        lp_x = self.base_distribution.log_prob(x)
        lp = lp_x - lc
        return lp

    def sample(self, key: PRNGKey, sample_shape=(1,)):
        z = self.base_distribution.sample(seed=key, sample_shape=sample_shape)
        y = jax.vmap(self.surjector.inverse)(z)
        return y

    def sample_and_log_prob(self, key: PRNGKey, sample_shape=(1,)):
        z, lp_z = self.base_distribution.sample_and_log_prob(
            seed=key, sample_shape=sample_shape
        )
        y, fldj = jax.vmap(self.surjector.forward_and_likelihood_contribution)(z)
        lp = jax.vmap(jnp.subtract)(lp_z, fldj)
        return y, lp
