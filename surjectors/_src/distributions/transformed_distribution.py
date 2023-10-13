import chex
import distrax
import haiku as hk
from distrax import Distribution
from jax import Array

from surjectors._src.surjectors.surjector import Surjector


class TransformedDistribution:
    """Distribution of a random variable transformed by a function.

    Can be used to define a pushforward measure.

    Examples:
        >>> import distrax
        >>> from jax import numpy as jnp
        >>> from surjectors import Slice, Chain, TransformedDistribution
        >>>
        >>> a = Slice(10)
        >>> b = Slice(5)
        >>> ab = Chain([a, b])
        >>>
        >>> TransformedDistribution(
        >>>     distrax.Normal(jnp.zeros(5), jnp.ones(5)),
        >>>     Chain([a, b])
        >>> )
    """

    def __init__(self, base_distribution: Distribution, transform: Surjector):
        """Constructs a TransformedDistribution.

        Args:
            base_distribution: a distribution object
            transform: some transformation
        """
        self.base_distribution = base_distribution
        self.transform = transform

    def __call__(self, method, **kwargs):
        """Call the TransformedDistribution object.

        Depending on "method", computes log-probability of an event or samples
        from the distribution.

        Args:
            method: either "sample" or "log_prob"
            **kwargs: several keyword arguments that are dispatched to
                whatever method is called.

        Returns:
            returns whatever 'method' returns
        """
        return getattr(self, method)(**kwargs)

    def log_prob(self, y: Array, x: Array = None):
        """Calculate the log probability of an event conditional on another.

        Args:
            y: event for which the log probability is computed
            x: optional event that is used to condition

        Returns:
            array of floats of log probabilities
        """
        _, lp = self.inverse_and_log_prob(y, x)
        return lp

    def inverse_and_log_prob(self, y: Array, x: Array = None):
        """Compute the inverse transformation and its log probability.

        Args:
            y: event for which the inverse and log probability is computed
            x: optional event that is used to condition

        Returns:
            tuple of two arrays of floats. The first one is the inverse
            transformation, the second one is the log probability
        """
        if x is not None:
            chex.assert_equal_rank([y, x])
            chex.assert_axis_dimension(y, 0, x.shape[0])

        if isinstance(self.transform, distrax.Bijector):
            z, lc = self.transform.inverse_and_log_det(y)
        else:
            z, lc = self.transform.inverse_and_likelihood_contribution(y, x=x)
        lp_z = self.base_distribution.log_prob(z)
        lp = lp_z + lc
        return z, lp

    def sample(self, sample_shape=(), x: Array = None):
        """Sample an event.

        Args:
            sample_shape: the size of the sample to be drawn
            x: optional event that is used to condition the samples.
                If x is given sample_shape is ignored

        Returns:
            a sample from the transformed distribution
        """
        y, _ = self.sample_and_log_prob(sample_shape, x)
        return y

    def sample_and_log_prob(self, sample_shape=(), x: Array = None):
        """Sample an event and compute its log probability.

        Args:
            sample_shape: the size of the sample to be drawn
            x: optional event that is used to condition the samples.
                If x is given sample_shape is ignored

        Returns:
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
        y, fldj = self.transform.forward_and_likelihood_contribution(z, x=x)
        lp = lp_z - fldj
        return y, lp
