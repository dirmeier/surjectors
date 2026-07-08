from surjectors._src._transform import Transform
from surjectors._src.bijectors.wrap import as_transform
from surjectors._src.surjectors.surjector import Surjector


class Chain(Surjector):
  """Chain of normalizing flows.

  Can be used to concatenate several normalizing flows together.

  Args:
      transforms: a list of transformations, such as bijections or
          surjections

  Examples:
      >>> from surjectors import Slice, Chain
      >>> a = Slice(10)
      >>> b = Slice(5)
      >>> ab = Chain([a, b])
  """

  def __init__(self, transforms: list[Transform]):
    self._transforms = [as_transform(t) for t in transforms]

  def _inverse_and_likelihood_contribution(self, y, x=None, **kwargs):
    z, lcs = self._transforms[0].inverse_and_likelihood_contribution(y, x=x)
    for transform in self._transforms[1:]:
      z, lc = transform.inverse_and_likelihood_contribution(z, x=x)
      lcs = lcs + lc
    return z, lcs

  def _forward_and_likelihood_contribution(self, z, x=None, **kwargs):
    y, log_det = self._transforms[-1].forward_and_likelihood_contribution(
      z, x=x
    )
    for transform in reversed(self._transforms[:-1]):
      y, lc = transform.forward_and_likelihood_contribution(y, x=x)
      log_det = log_det + lc
    return y, log_det
