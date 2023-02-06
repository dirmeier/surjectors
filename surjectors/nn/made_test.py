# pylint: skip-file

import chex
import distrax
import haiku as hk
import jax
import optax
import pytest
from jax import numpy as jnp
from jax import random
from tensorflow_probability.substrates.jax.bijectors.masked_autoregressive import (
    AutoregressiveNetwork,
)

import surjectors
from surjectors.conditioners.mlp import mlp_conditioner
from surjectors.nn.made import MADE
from surjectors.util import make_alternating_binary_mask, unstack


@pytest.fixture()
def flow(request):
    @hk.without_apply_rng
    @hk.transform
    def _flow(**kwargs):
        made = MADE(
            request.param, [2], 2, w_init=jnp.ones, activation=lambda x: x
        )
        return made(**kwargs)

    return _flow


@pytest.mark.parametrize("flow", [3], indirect=True)
def test_made_shape(flow):
    x = jnp.ones((1, 3))
    params = flow.init(random.PRNGKey(0), y=x)
    y = flow.apply(params, y=x)
    chex.assert_shape(y, (1, 3, 2))


# @pytest.mark.parametrize('flow', [3], indirect=True)
# def test_made_output(flow):
#     x = jnp.ones((1, 3))
#     params = flow.init(random.PRNGKey(0), y=x)
#     y = flow.apply(params, y=x)
#     a, b = unstack(y, -1)
#     chex.assert_trees_all_equal(a, jnp.array([[0.0, 1.0, 3.0]]))
#     chex.assert_trees_all_equal(b, jnp.array([[0.0, 1.0, 3.0]]))


@pytest.mark.parametrize("flow", [3], indirect=True)
def test_conditional_made_output(flow):
    y = jnp.ones((1, 3))
    x = jnp.ones((1, 2))
    params = flow.init(random.PRNGKey(0), y=y, x=x)
    y = flow.apply(params, y=y, x=x)
    a, b = unstack(y, -1)
    chex.assert_trees_all_equal(a, jnp.array([[0.0, 3.0, 7.0]]))
    chex.assert_trees_all_equal(b, jnp.array([[0.0, 3.0, 7.0]]))

    ma = AutoregressiveNetwork(
        2,
        3,
        True,
        2,
        hidden_units=[2],
        conditional_input_layers="first_layer",
        kernel_initializer=lambda x: 1,
    )
    s = ma(y, x)
    print(s)
