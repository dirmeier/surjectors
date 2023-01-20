from typing import Tuple

import chex
import distrax
import haiku as hk
import jax
import jax.numpy as jnp
from chex import Array
from distrax import Distribution

from surjectors.surjectors.surjector import Surjector


class TransformedDistribution:
    """
    Distribution of a random variable transformed by a surjective or
    bijectiive function
    """

    def __init__(self, base_distribution: Distribution, surjector: Surjector):
        self.base_distribution = base_distribution
        self.surjector = surjector

    def __call__(self, method, **kwargs):
        return getattr(self, method)(**kwargs)

    def log_prob(self, y: Array, x: Array = None) -> Array:
        """
        Calculate the log probabilisty of an event conditional on another event

        Parameters
        ----------
        y: jnp.ndarray
            event for which the log probability is computed
        x: Optional[jnp.ndarray]
            optional event that is used to condition

        Returns
        -------
        jnp.ndarray
            array of floats of log probabilities
        """

        _, lp = self.inverse_and_log_prob(y, x)
        return lp

    def inverse_and_log_prob(
        self, y: Array, x: Array = None
    ) -> Tuple[Array, Array]:
        if x is not None:
            chex.assert_equal_rank([y, x])
            chex.assert_axis_dimension(y, 0, x.shape[0])
            chex.assert_axis_dimension(y, 1, x.shape[1])

        if isinstance(self.surjector, distrax.Bijector):
            z, lc = self.surjector.inverse_and_log_det(y)
        else:
            z, lc = self.surjector.inverse_and_likelihood_contribution(y, x=x)
        lp_z = self.base_distribution.log_prob(z)
        lp = lp_z + lc
        return z, lp

    def sample(self, sample_shape=(), x: Array = None):
        if x is not None and len(sample_shape):
            chex.assert_equal(sample_shape[0], x.shape[0])
        elif x is not None:
            sample_shape = (x.shape[0],)
        z = self.base_distribution.sample(
            seed=hk.next_rng_key(), sample_shape=sample_shape
        )
        y = jax.vmap(self.surjector.forward)(z, x)
        return y

    def sample_and_log_prob(self, sample_shape=(1,), x: Array = None):
        z, lp_z = self.base_distribution.sample_and_log_prob(
            seed=hk.next_rng_key(), sample_shape=sample_shape, x=x
        )
        y, fldj = jax.vmap(self.surjector.forward_and_likelihood_contribution)(
            z, x=x
        )
        lp = jax.vmap(jnp.subtract)(lp_z, fldj)
        return y, lp
