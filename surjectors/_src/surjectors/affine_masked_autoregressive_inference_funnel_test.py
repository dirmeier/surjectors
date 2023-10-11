# pylint: skip-file

import distrax
import haiku as hk
from jax import numpy as jnp
from jax import random

from surjectors import TransformedDistribution
from surjectors.conditioners import MADE
from surjectors.conditioners.mlp import mlp_conditioner
from surjectors.surjectors.affine_masked_autoregressive_inference_funnel import (  # noqa: E501
    AffineMaskedAutoregressiveInferenceFunnel,
)


def _decoder_fn(n_dim):
    decoder_net = mlp_conditioner([4, 4, n_dim * 2])

    def _fn(z):
        params = decoder_net(z)
        mu, log_scale = jnp.split(params, 2, -1)
        return distrax.Independent(distrax.Normal(mu, jnp.exp(log_scale)))

    return _fn


def _base_distribution_fn(n_latent):
    base_distribution = distrax.Independent(
        distrax.Normal(jnp.zeros(n_latent), jnp.ones(n_latent)),
        reinterpreted_batch_ndims=1,
    )
    return base_distribution


def _get_funnel_surjector(n_latent, n_dimension):
    return AffineMaskedAutoregressiveInferenceFunnel(
        n_latent,
        _decoder_fn(n_dimension - n_latent),
        MADE(n_latent, [4, 4], 2),
    )


def make_surjector(n_dimension, n_latent):
    def _transformation_fn(n_dimension):
        funnel = _get_funnel_surjector(n_latent, n_dimension)
        return funnel

    def _flow(method, **kwargs):
        td = TransformedDistribution(
            _base_distribution_fn(n_latent), _transformation_fn(n_dimension)
        )
        return td(method, **kwargs)

    td = hk.transform(_flow)
    return td


def test_affine_masked_autoregressive_inference_funnel():
    n_dimension, n_latent = 4, 2
    y = random.normal(random.PRNGKey(1), shape=(10, n_dimension))

    flow = make_surjector(n_dimension, n_latent)
    params = flow.init(random.PRNGKey(0), method="log_prob", y=y)
    _ = flow.apply(params, None, method="log_prob", y=y)


def test_conditional_affine_masked_autoregressive_inference_funnel():
    n_dimension, n_latent = 4, 2
    y = random.normal(random.PRNGKey(1), shape=(10, n_dimension))
    x = random.normal(random.PRNGKey(1), shape=(10, 2))

    flow = make_surjector(n_dimension, n_latent)
    params = flow.init(random.PRNGKey(0), method="log_prob", y=y, x=x)
    _ = flow.apply(params, None, method="log_prob", y=y, x=x)
