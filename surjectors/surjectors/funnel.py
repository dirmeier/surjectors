from abc import ABC

from surjectors.surjectors.surjector import Surjector


# pylint: disable=too-many-arguments
class Funnel(Surjector, ABC):
    """
    Abstract funnel base class
    """

    def __init__(self, n_keep, decoder, conditioner, encoder, kind):
        super().__init__(n_keep, decoder, encoder, kind)
        self._conditioner = conditioner
