from abc import ABC

from surjectors.surjectors.surjector import Surjector


class Funnel(Surjector, ABC):
    def __init__(self, n_keep, decoder, conditioner, encoder, kind):
        super().__init__(n_keep, decoder, encoder, kind)
        self._conditioner = conditioner
