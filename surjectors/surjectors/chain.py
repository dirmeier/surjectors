from surjectors.surjectors.surjector import Surjector


class Chain(Surjector):
    def __init__(self, surjectors):
        super().__init__(None, None, None, "surjector")
        self._surjectors = surjectors

    def inverse_and_likelihood_contribution(self, y, x=None):
        z, lcs = self._inverse_and_log_contribution_dispatch(
            self._surjectors[0], y, x
        )
        for surjector in self._surjectors[1:]:
            z, lc = self._inverse_and_log_contribution_dispatch(surjector, z, x)
            lcs += lc
        return z, lcs

    @staticmethod
    def _inverse_and_log_contribution_dispatch(surjector, y, x):
        if isinstance(surjector, Surjector):
            fn = getattr(surjector, "inverse_and_likelihood_contribution")
            z, lc = fn(y, x)
        else:
            # TODO: not correct this
            fn = getattr(surjector, "inverse_and_log_det")
            z, lc = fn(y, x)
        return z, lc

    def forward_and_likelihood_contribution(self, z, x=None):
        y, log_det = self._surjectors[-1].forward_and_log_det(z, x)
        for surjector in reversed(self._surjectors[:-1]):
            y, lc = self._forward_and_log_contribution_dispatch(surjector, y, x)
            log_det += lc
        return y, log_det

    @staticmethod
    def _forward_and_log_contribution_dispatch(surjector, y, x):
        if isinstance(surjector, Surjector):
            fn = getattr(surjector, "forward_and_likelihood_contribution")
            z, lc = fn(y, x)
        else:
            fn = getattr(surjector, "forward_and_log_det")
            z, lc = fn(y)
        return z, lc

    def forward(self, z, x=None):
        y, _ = self.forward_and_likelihood_contribution(z, x)
        return y
