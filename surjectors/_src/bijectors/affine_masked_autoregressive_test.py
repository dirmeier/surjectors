# pylint: skip-file

import distrax
import haiku as hk
from jax import numpy as jnp
from jax import random

from surjectors import AffineMaskedAutoregressive, TransformedDistribution
from surjectors.nn import MADE


def _base_distribution_fn(n_latent):
    base_distribution = distrax.Independent(
        distrax.Normal(jnp.zeros(n_latent), jnp.ones(n_latent)),
        reinterpreted_batch_ndims=1,
    )
    return base_distribution


def make_bijector(n_dimension):
    def _transformation_fn(n_dimension):
        bij = AffineMaskedAutoregressive(
            MADE(n_dimension, [8, 8], 2),
        )
        return bij

    def _flow(method, **kwargs):
        td = TransformedDistribution(
            _base_distribution_fn(n_dimension), _transformation_fn(n_dimension)
        )
        return td(method, **kwargs)

    td = hk.transform(_flow)
    return td


def test_affine_masked_autoregressive():
    n_dimension = 4
    y = random.normal(random.PRNGKey(1), shape=(10, n_dimension))

    flow = make_bijector(n_dimension)
    params = flow.init(random.PRNGKey(0), method="log_prob", y=y)
    _ = flow.apply(params, None, method="log_prob", y=y)


def test_conditional_affine_masked_autoregressive():
    n_dimension = 4
    y = random.normal(random.PRNGKey(1), shape=(10, n_dimension))
    x = random.normal(random.PRNGKey(1), shape=(10, 2))

    flow = make_bijector(n_dimension)
    params = flow.init(random.PRNGKey(0), method="log_prob", y=y, x=x)
    _ = flow.apply(params, None, method="log_prob", y=y, x=x)
