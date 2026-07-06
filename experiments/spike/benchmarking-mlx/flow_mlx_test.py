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
