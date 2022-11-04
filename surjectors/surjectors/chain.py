from surjectors.surjectors.surjector import Surjector


class Chain(Surjector):
    def __init__(self, surjectors):
        super().__init__(None, None, None, "surjector")
        self._surjectors = surjectors

    def inverse_and_likelihood_contribution(self, y):
        z, log_det = self._surjectors[0].forward_and_log_det(y)
        for surjector in self._surjectors[1:]:
            x, lc = surjector.inverse_and_likelihood_contribution(z)
            log_det += lc
        return z, log_det

    def forward_and_likelihood_contribution(self, z):
        y, log_det = self._surjectors[-1].forward_and_log_det(z)
        for _surjectors in reversed(self._surjectors[:-1]):
            x, lc = _surjectors.forward_and_log_det(x)
            log_det += lc
        return y, log_det
