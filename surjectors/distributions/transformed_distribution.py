from typing import Tuple

import chex
import distrax
import haiku as hk
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
        Calculate the log probability of an event conditional on another event

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
        """
        Compute the inverse transformation and the log probability of an event
        conditional on another event

        Parameters
        ----------
        y: jnp.ndarray
            event for which the inverse and log probability is computed
        x: Optional[jnp.ndarray]
            optional event that is used to condition

        Returns
        -------
        Tuple[jnp.ndarray, jnp.ndarray]
            tuple of two arrays of floats. The first one is the inverse
            transformation, the second one is the log probability
        """

        if x is not None:
            chex.assert_equal_rank([y, x])
            chex.assert_axis_dimension(y, 0, x.shape[0])

        if isinstance(self.surjector, distrax.Bijector):
            z, lc = self.surjector.inverse_and_log_det(y)
        else:
            z, lc = self.surjector.inverse_and_likelihood_contribution(y, x=x)
        lp_z = self.base_distribution.log_prob(z)
        lp = lp_z + lc
        return z, lp

    def sample(self, sample_shape=(), x: Array = None):
        """
        Sample an event

        Parameters
        ----------
        sample_shape: Tuple[int]
            the size of the sample to be drawn
        x: Optional[jnp.ndarray]
            optional event that is used to condition the samples. If x is given
            sample_shape is ignored

        Returns
        -------
        jnp.ndarray
            a sample from the transformed distribution
        """

        y, _ = self.sample_and_log_prob(sample_shape, x)
        return y

    def sample_and_log_prob(self, sample_shape=(), x: Array = None):
        """
        Sample an event and compute its log probability

        Parameters
        ----------
        sample_shape: Tuple[int]
            the size of the sample to be drawn
        x: Optional[jnp.ndarray]
            optional event that is used to condition the samples. If x is given
            sample_shape is ignored

        Returns
        -------
        Tuple[jnp.ndarray, jnp.ndarray]
            tuple of two arrays of floats. The first one is the drawn sample
            transformation, the second one is its log probability
        """

        if x is not None and len(sample_shape) == 0:
            sample_shape = (x.shape[0],)
        if x is not None:
            chex.assert_equal(sample_shape[0], x.shape[0])

        z, lp_z = self.base_distribution.sample_and_log_prob(
            seed=hk.next_rng_key(),
            sample_shape=sample_shape,
        )
        y, fldj = self.surjector.forward_and_likelihood_contribution(z, x=x)
        lp = lp_z - fldj
        return y, lp
