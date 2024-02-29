from typing import Callable, Optional

import distrax

from surjectors._src.bijectors.masked_coupling import MaskedCoupling
from surjectors._src.distributions.transformed_distribution import Array


# pylint: disable=too-many-arguments, arguments-renamed,too-many-ancestors
class RationalQuadraticSplineMaskedCoupling(MaskedCoupling):
    """A rational quadratic spline masked coupling layer.

    References:
        .. [1] Dinh, Laurent, et al. "Density estimation using RealNVP".
            International Conference on Learning Representations, 2017.
        .. [2] Durkan, Conor, et al. "Neural Spline Flows".
            Advances in Neural Information Processing Systems, 2019.

    Examples:
        >>> import distrax
        >>> from surjectors import RationalQuadraticSplineMaskedCoupling
        >>> from surjectors.nn import make_mlp
        >>> from surjectors.util import make_alternating_binary_mask
        >>>
        >>> layer = RationalQuadraticSplineMaskedCoupling(
        >>>     mask=make_alternating_binary_mask(10, True),
        >>>     conditioner=make_mlp([8, 8, 10 * 2]),
        >>>     range_min=-1.0,
        >>>     range_max=1.0
        >>> )
    """

    def __init__(
        self,
        mask: Array,
        conditioner: Callable,
        range_min: float,
        range_max: float,
        event_ndims: Optional[int] = None,
        inner_event_ndims: int = 0,
    ):
        """Construct a rational quadratic spline masked coupling layer.

        Args:
            mask: a boolean mask of length n_dim. A value
                of True indicates that the corresponding input remains unchanged
            conditioner: a function that computes the parameters of the inner
                bijector
            range_min: minimum range of the spline
            range_max: maximum range of the spline
            event_ndims: the number of array dimensions the bijector operates on
            inner_event_ndims: the number of array dimensions the inner bijector
                operates on
        """
        self.range_min = range_min
        self.range_max = range_max

        def _bijector_fn(params: Array):
            return distrax.RationalQuadraticSpline(
                params, self.range_min, self.range_max
            )

        super().__init__(
            mask, conditioner, _bijector_fn, event_ndims, inner_event_ndims
        )
