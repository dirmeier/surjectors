# pylint: skip-file

import chex
import distrax
import haiku as hk
import jax
import optax
import pytest
from jax import numpy as jnp
from jax import random

import surjectors
from surjectors import TransformedDistribution
from surjectors.bijectors.masked_autoregressive import MaskedAutoregressive
from surjectors.nn.made import MADE
from surjectors.util import unstack


def _affine_bijector_fn(params):
    means, log_scales = unstack(params, -1)
    return distrax.Inverse(distrax.ScalarAffine(means, jnp.exp(log_scales)))


def _rq_bijector_fn(params):
    return distrax.Inverse(distrax.RationalQuadraticSpline(params, -2.0, 2.0))


def _base_distribution_fn(n_latent):
    base_distribution = distrax.Independent(
        distrax.Normal(jnp.zeros(n_latent), jnp.ones(n_latent)),
        reinterpreted_batch_ndims=1,
    )
    return base_distribution


def masked_autoregressive_bijector(n_dim, bijector_fn, n_params, n_hidden):
    def _transformation_fn(n_dim):
        layer = MaskedAutoregressive(
            bijector_fn=bijector_fn,
            conditioner=MADE(
                n_dim,
                [n_hidden],
                n_params,
                w_init=hk.initializers.TruncatedNormal(stddev=1.0),
                b_init=jnp.ones,
            ),
        )

        return layer

    def _flow(**kwargs):
        td = TransformedDistribution(
            _base_distribution_fn(n_dim), _transformation_fn(n_dim)
        )
        return td.log_prob(**kwargs)

    td = hk.transform(_flow)
    td = hk.without_apply_rng(td)
    return td


@pytest.fixture(
    params=[(_rq_bijector_fn, 4), (_affine_bijector_fn, 2)],
    ids=["rq_masked_autoregressive", "affine_masked_autoregressive"],
)
def bijection(request):
    yield request.param


def test_unconditional_bijector(bijection):
    rng_seq = hk.PRNGSequence(0)

    bijector_fn, n_params = bijection
    n_dim, n_hidden = 2, 8
    flow = masked_autoregressive_bijector(2, bijector_fn, n_params, n_hidden)
    y = distrax.Normal(0.0, 1.0).sample(
        seed=random.PRNGKey(2), sample_shape=(100, n_dim)
    )

    params = flow.init(next(rng_seq), y=y)
    chex.assert_shape(
        params["made/~/masked_linear_1"]["w"], (8, n_dim * n_params)
    )


def test_conditional_bijector(bijection):
    rng_seq = hk.PRNGSequence(0)

    bijector_fn, n_params = bijection
    n_dim, n_hidden = 2, 8
    flow = masked_autoregressive_bijector(2, bijector_fn, n_params, n_hidden)
    y = distrax.Normal(0.0, 1.0).sample(
        seed=random.PRNGKey(2), sample_shape=(100, n_dim)
    )
    x = distrax.Normal(0.0, 1.0).sample(
        seed=random.PRNGKey(2), sample_shape=(100, 1)
    )

    params = flow.init(next(rng_seq), y=y, x=x)
    chex.assert_shape(
        params["made/~/masked_linear_1"]["w"], (8, n_dim * n_params)
    )
