# pylint: skip-file

import distrax
import haiku as hk
from jax import numpy as jnp
from jax import random

from surjectors import Slice, TransformedDistribution
from surjectors._src.conditioners.mlp import make_mlp


def _decoder_fn(n_dim):
    def _fn(z):
        params = make_mlp([32, 32, n_dim * 2])(z)
        means, log_scales = jnp.split(params, 2, -1)
        return distrax.Independent(distrax.Normal(means, jnp.exp(log_scales)))

    return _fn


def _base_distribution_fn(n_latent):
    base_distribution = distrax.Independent(
        distrax.Normal(jnp.zeros(n_latent), jnp.ones(n_latent)),
        reinterpreted_batch_ndims=1,
    )
    return base_distribution


def make_surjector(n_dimension, n_latent):
    def _transformation_fn():
        slice = Slice(n_latent, _decoder_fn(n_dimension - n_latent))
        return slice

    def _flow(method, **kwargs):
        td = TransformedDistribution(
            _base_distribution_fn(n_latent), _transformation_fn()
        )
        return td(method, **kwargs)

    td = hk.transform(_flow)
    return td


def test_slice():
    n_dimension, n_latent = 10, 3
    y = random.normal(random.PRNGKey(1), shape=(10, n_dimension))

    flow = make_surjector(n_dimension, n_latent)
    params = flow.init(random.PRNGKey(0), method="log_prob", y=y)
    _ = flow.apply(params, None, method="log_prob", y=y)


def test_conditional_slice():
    n_dimension, n_latent = 10, 3
    y = random.normal(random.PRNGKey(1), shape=(10, n_dimension))
    x = random.normal(random.PRNGKey(1), shape=(10, 2))

    flow = make_surjector(n_dimension, n_latent)
    params = flow.init(random.PRNGKey(0), method="log_prob", y=y, x=x)
    _ = flow.apply(params, None, method="log_prob", y=y, x=x)
