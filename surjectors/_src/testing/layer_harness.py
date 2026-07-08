"""Test harness for exercising a single layer's transform math in isolation."""

import haiku as hk


def transform_layer(build_layer, direction: str):
  """Haiku-transform a single layer's forward or inverse math.

  Args:
      build_layer: zero-argument callable constructing the layer inside the
          transform (so Haiku modules bind correctly)
      direction: "forward" or "inverse"

  Returns:
      an ``(init, apply)`` pair. ``init(rng, z_or_y, x=None)`` returns
      params. ``apply(params, z_or_y, x=None, rng=None)`` returns
      ``(output, likelihood_contribution)``; pass ``rng`` for layers whose
      math samples internally (e.g. a funnel's stochastic forward pass).
  """
  if direction not in ("forward", "inverse"):
    raise ValueError(
      f"direction must be 'forward' or 'inverse', got {direction!r}"
    )

  def _fn(z_or_y, x=None):
    layer = build_layer()
    if direction == "forward":
      return layer._forward_and_likelihood_contribution(z_or_y, x=x)
    return layer._inverse_and_likelihood_contribution(z_or_y, x=x)

  transformed = hk.transform(_fn)

  def init(rng, z_or_y, x=None):
    return transformed.init(rng, z_or_y, x=x)

  def apply(params, z_or_y, x=None, rng=None):
    return transformed.apply(params, rng, z_or_y, x=x)

  return init, apply
