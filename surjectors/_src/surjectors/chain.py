from typing import List

from surjectors._src._transform import Transform
from surjectors._src.surjectors.surjector import Surjector


class Chain(Surjector):
    """Chain of normalizing flows.

    Can be used to concatenate several normalizing flows together.

    Examples:
        >>> from surjectors import Slice, Chain
        >>> a = Slice(10)
        >>> b = Slice(5)
        >>> ab = Chain([a, b])
    """

    def __init__(self, transforms: List[Transform]):
        """Constructs a Chain.

        Args:
            transforms: a list of transformations, such as bijections or
                surjections
        """
        self._transforms = transforms

    def _inverse_and_likelihood_contribution(self, y, x=None, **kwargs):
        z, lcs = self._inverse_and_log_contribution_dispatch(
            self._transforms[0], y, x
        )
        for transform in self._transforms[1:]:
            z, lc = self._inverse_and_log_contribution_dispatch(transform, z, x)
            lcs += lc
        return z, lcs

    @staticmethod
    def _inverse_and_log_contribution_dispatch(transform, y, x):
        if isinstance(transform, Surjector):
            if hasattr(transform, "inverse_and_likelihood_contribution"):
                fn = getattr(transform, "inverse_and_likelihood_contribution")
            else:
                fn = getattr(transform, "inverse_and_log_det")
            z, lc = fn(y, x)
        else:
            fn = getattr(transform, "inverse_and_log_det")
            z, lc = fn(y)
        return z, lc

    def _forward_and_likelihood_contribution(self, z, x=None, **kwargs):
        y, log_det = self._forward_and_log_contribution_dispatch(
            self._transforms[-1], z, x
        )
        for transform in reversed(self._transforms[:-1]):
            y, lc = self._forward_and_log_contribution_dispatch(transform, y, x)
            log_det += lc
        return y, log_det

    @staticmethod
    def _forward_and_log_contribution_dispatch(transform, z, x):
        if isinstance(transform, Surjector):
            if hasattr(transform, "forward_and_likelihood_contribution"):
                fn = getattr(transform, "forward_and_likelihood_contribution")
            else:
                fn = getattr(transform, "forward_and_log_det")
            y, lc = fn(z, x)
        else:
            fn = getattr(transform, "forward_and_log_det")
            y, lc = fn(z)
        return y, lc
