from surjectors.surjectors.surjector import Surjector


class Chain(Surjector):
    def __init__(self, surjectors):
        super().__init__(None, None, None, "surjector")
        self._surjectors = surjectors

    def inverse_and_likelihood_contribution(self, y):
        z, lcs = self._inverse_and_log_contribution_dispatch(
            self._surjectors[0], y
        )
        for surjector in self._surjectors[1:]:
            z, lc = self._inverse_and_log_contribution_dispatch(surjector, z)
            lcs += lc
        return z, lcs

    @staticmethod
    def _inverse_and_log_contribution_dispatch(surjector, y):
        if isinstance(surjector, Surjector):
            fn = getattr(surjector, "inverse_and_likelihood_contribution")
        else:
            fn = getattr(surjector, "inverse_and_log_det")
        z, lc = fn(y)
        return z, lc

    def forward_and_likelihood_contribution(self, z):
        y, log_det = self._surjectors[-1].forward_and_log_det(z)
        for _surjectors in reversed(self._surjectors[:-1]):
            x, lc = _surjectors.forward_and_log_det(x)
            log_det += lc
        return y, log_det
