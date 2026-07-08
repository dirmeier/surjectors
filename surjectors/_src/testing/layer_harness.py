"""Test harness for exercising a single layer's transform math in isolation."""

import haiku as hk


def transform_layer(build_layer, direction: str) -> hk.Transformed:
  """Haiku-transform a single layer's forward or inverse math.

  Args:
      build_layer: zero-argument callable constructing the layer inside the
          transform (so Haiku modules bind correctly)
      direction: "forward" or "inverse"

  Returns:
      an ``hk.Transformed`` pair; ``apply(params, z_or_y, x=None)`` returns
      ``(output, likelihood_contribution)``
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

  return hk.without_apply_rng(hk.transform(_fn))
