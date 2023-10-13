from abc import ABCMeta

from distrax._src.utils import jittable


class Transform(jittable.Jittable, metaclass=ABCMeta):
    """Transformation of a random variable."""
