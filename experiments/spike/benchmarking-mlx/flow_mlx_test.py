import math

import mlx.core as mx
import numpy as np
from scipy.stats import norm

from flow_mlx import DiagNormal


def test_diagnormal_matches_analytic():
  z = mx.array([[0.0, 0.0], [1.0, -2.0], [0.5, 3.0]])
  got = np.array(DiagNormal(dim=2).log_prob(z))
  want = norm.logpdf(np.array(z)).sum(axis=-1)
  np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


import mlx.nn as nn

from flow_mlx import MADE


def test_made_is_autoregressive():
  # Output params for dimension d must depend only on inputs < d.
  made = MADE(input_size=3, hidden_sizes=(16, 16), n_params=2)
  x = mx.array([[0.3, -1.2, 2.0]])

  def out_d(x, d):
    # sum over the n_params of output dimension d
    return mx.sum(made(x)[:, d, :])

  for d in range(3):
    grad = np.array(mx.grad(out_d)(x, d))[0]
    # gradient wrt inputs >= d must be exactly zero
    assert np.allclose(grad[d:], 0.0), (d, grad)
    if d > 0:
      # at least one earlier input should influence output d
      assert not np.allclose(grad[:d], 0.0), (d, grad)
