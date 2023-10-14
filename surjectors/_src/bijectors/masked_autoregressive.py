from typing import Callable

from distrax._src.utils import math
from jax import numpy as jnp

from surjectors._src.bijectors.bijector import Bijector
from surjectors._src.conditioners.nn.made import MADE


# pylint: disable=too-many-arguments, arguments-renamed
class MaskedAutoregressive(Bijector):
    """A masked autoregressive layer.

    References:
        .. [1] Papamakarios, George, et al. "Masked Autoregressive Flow for
            Density Estimation". Advances in Neural Information Processing
            Systems, 2017.

    Examples:
        >>> import distrax
        >>> from surjectors import MaskedAutoregressive
        >>> from surjectors.util import unstack
        >>>
        >>> def bijector_fn(params):
        >>>     means, log_scales = unstack(params, -1)
        >>>     return distrax.ScalarAffine(means, jnp.exp(log_scales))
        >>>
        >>> layer = MaskedAutoregressive(
        >>>     conditioner=MADE(10, [8, 8], 2),
        >>>     bijector_fn=bijector_fn
        >>> )
    """

    def __init__(
        self,
        conditioner: MADE,
        bijector_fn: Callable,
        event_ndims: int = 1,
        inner_event_ndims: int = 0,
    ):
        """Construct a masked autoregressive layer.

        Args:
            conditioner: a MADE network
            bijector_fn: a callable that returns the inner bijector that will
                be used to transform the input
            event_ndims: the number of array dimensions the bijector operates on
            inner_event_ndims: tthe number of array dimensions the bijector
                operates on
        """
        if event_ndims is not None and event_ndims < inner_event_ndims:
            raise ValueError(
                f"`event_ndims={event_ndims}` should be at least as"
                f" large as `inner_event_ndims={inner_event_ndims}`."
            )
        if not isinstance(conditioner, MADE):
            raise ValueError(
                "conditioner should be a MADE when used "
                "MaskedAutoregressive flow"
            )
        self._event_ndims = event_ndims
        self._inner_event_ndims = inner_event_ndims
        self.conditioner = conditioner
        self._inner_bijector = bijector_fn

    def _forward_and_likelihood_contribution(self, z, x=None, **kwargs):
        y = jnp.zeros_like(z)
        for _ in jnp.arange(z.shape[-1]):
            params = self.conditioner(y, x)
            y, log_det = self._inner_bijector(params).forward_and_log_det(z)
        log_det = math.sum_last(
            log_det, self._event_ndims - self._inner_event_ndims
        )
        return y, log_det

    def _inverse_and_likelihood_contribution(self, y, x=None, **kwargs):
        params = self.conditioner(y, x)
        z, log_det = self._inner_bijector(params).inverse_and_log_det(y)
        log_det = math.sum_last(
            log_det, self._event_ndims - self._inner_event_ndims
        )
        return z, log_det
