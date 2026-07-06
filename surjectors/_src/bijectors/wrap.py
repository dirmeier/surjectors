"""Adapt foreign bijectors to the surjectors transform interface."""

from surjectors._src.bijectors.bijector import Bijector
from surjectors._src.surjectors.surjector import Surjector


class _BijectorAdapter(Bijector):
  """Present a distrax-style bijector as a surjectors transform.

  A distrax bijector exposes ``forward_and_log_det`` / ``inverse_and_log_det``
  without conditioning. This adapter re-expresses it in the surjectors
  likelihood-contribution interface so composition code (``Chain``,
  ``TransformedDistribution``) has a single seam to call and never has to know
  whether a transform is native or foreign.
  """

  def __init__(self, bijector):
    self._bijector = bijector

  def _forward_and_likelihood_contribution(self, z, x=None, **kwargs):
    return self._bijector.forward_and_log_det(z)

  def _inverse_and_likelihood_contribution(self, y, x=None, **kwargs):
    return self._bijector.inverse_and_log_det(y)


def as_transform(transform) -> Surjector:
  """Return a surjectors transform, wrapping a foreign bijector if needed.

  Args:
      transform: either a native surjectors transform or a distrax-style
          bijector exposing ``forward_and_log_det`` / ``inverse_and_log_det``

  Returns:
      ``transform`` unchanged if it is already a native transform, otherwise
      a ``_BijectorAdapter`` wrapping it
  """
  if isinstance(transform, Surjector):
    return transform
  return _BijectorAdapter(transform)
