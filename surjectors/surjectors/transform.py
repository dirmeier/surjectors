from abc import ABCMeta, abstractmethod

import distrax
from distrax._src.utils import jittable


class Transform(jittable.Jittable, metaclass=ABCMeta):
    """
    A transformation of a random variable
    """

    @abstractmethod
    def inverse_and_likelihood_contribution(self, y):
        pass

    @abstractmethod
    def forward_and_likelihood_contribution(self, z):
        pass
