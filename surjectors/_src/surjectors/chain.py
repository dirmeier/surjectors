from typing import List

from surjectors._src.surjectors._transform import Transform
from surjectors._src.surjectors.surjector import Surjector


class Chain(Surjector):
    """
    Chain of normalizing flows.

    Can be used to concatenate several normalizing flows together.

    Examples:

        >>> from surjectors import Slice, Chain
        >>> a = Slice(10)
        >>> b = Slice(5)
        >>> ab = Chain([a, b])
    """

    def __init__(self, transforms: List[Transform]):
        """
        Constructs a Chain.

        Args:
            transforms: a list of transformations, such as bijections or
                surjections
        """

        super().__init__(None, None, None, "surjector")
        self._transforms = transforms

    def inverse_and_likelihood_contribution(self, y, x=None, **kwargs):
        z, lcs = self._inverse_and_log_contribution_dispatch(
            self._transforms[0], y, x
        )
        for surjector in self._transforms[1:]:
            z, lc = self._inverse_and_log_contribution_dispatch(surjector, z, x)
            lcs += lc
        return z, lcs

    @staticmethod
    def _inverse_and_log_contribution_dispatch(surjector, y, x):
        if isinstance(surjector, Surjector):
            if hasattr(surjector, "inverse_and_likelihood_contribution"):
                fn = getattr(surjector, "inverse_and_likelihood_contribution")
            else:
                fn = getattr(surjector, "inverse_and_log_det")
            z, lc = fn(y, x)
        else:
            fn = getattr(surjector, "inverse_and_log_det")
            z, lc = fn(y)
        return z, lc

    def forward_and_likelihood_contribution(self, z, x=None, **kwargs):
        y, log_det = self._forward_and_log_contribution_dispatch(
            self._transforms[-1], z, x
        )
        for surjector in reversed(self._transforms[:-1]):
            y, lc = self._forward_and_log_contribution_dispatch(surjector, y, x)
            log_det += lc
        return y, log_det

    @staticmethod
    def _forward_and_log_contribution_dispatch(surjector, z, x):
        if isinstance(surjector, Surjector):
            if hasattr(surjector, "forward_and_likelihood_contribution"):
                fn = getattr(surjector, "forward_and_likelihood_contribution")
            else:
                fn = getattr(surjector, "forward_and_log_det")
            y, lc = fn(z, x)
        else:
            fn = getattr(surjector, "forward_and_log_det")
            y, lc = fn(z)
        return y, lc

    def forward(self, z, x=None):
        y, _ = self.forward_and_likelihood_contribution(z, x)
        return y
