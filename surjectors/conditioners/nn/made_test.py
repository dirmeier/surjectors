# pylint: skip-file

import chex
import haiku as hk
import pytest
from jax import numpy as jnp
from jax import random

from surjectors.conditioners import MADE
from surjectors.util import unstack


@pytest.fixture()
def made(request):

    input_shape, hidden_shapes, output_shape = request.param

    @hk.without_apply_rng
    @hk.transform
    def _made(**kwargs):
        made = MADE(
            input_shape,
            hidden_shapes,
            output_shape,
            w_init=jnp.ones,
            activation=lambda x: x,
        )
        return made(**kwargs)

    return _made


@pytest.mark.parametrize("made", [(3, [2, 4, 2], 2)], indirect=True)
def test_made_shape(made):
    x = jnp.ones((1, 3))
    params = made.init(random.PRNGKey(0), y=x)
    y = made.apply(params, y=x)
    chex.assert_shape(y, (1, 3, 2))


@pytest.mark.parametrize(
    "made,expected",
    [
        ((3, [2, 4, 2], 2), [[0.0, 2.0, 10.0]]),
        ((3, [2], 2), [[0.0, 1.0, 3.0]]),
    ],
    indirect=["made"],
)
def test_made_output(made, expected):
    """
    Tested using:

    y = np.ones((1, 3))
    made = AutoregressiveNetwork(
        2,
        3,
        hidden_units=[2, 4, 2], # or [2]
        conditional_input_layers="first_layer",
        kernel_initializer="ones",
        activation=lambda x: x
    )
    y = made(y)
    a, b = unstack(y, -1)
    """

    y = jnp.ones((1, 3))
    params = made.init(random.PRNGKey(0), y=y)
    y = made.apply(params, y=y)
    a, b = unstack(y, -1)
    chex.assert_trees_all_equal(a, jnp.array(expected))
    chex.assert_trees_all_equal(b, jnp.array(expected))


@pytest.mark.parametrize(
    "made,expected",
    [
        ((3, [2, 4, 2], 2), [[0.0, 6.0, 26.0]]),
        ((3, [2], 2), [[0.0, 3.0, 7.0]]),
    ],
    indirect=["made"],
)
def test_conditional_made_output(made, expected):
    """
    Tested using:

    y = np.ones((1, 3))
    x = np.ones((1, 2))
    made = AutoregressiveNetwork(
      2,
      3,
      True,
      2,
      hidden_units=[2, 4, 2], # or [2]
      conditional_input_layers="first_layer",
      kernel_initializer="ones",
      activation=lambda x: x
    )
    y = made(y, x)
    a, b = unstack(y, -1)
    """

    y = jnp.ones((1, 3))
    x = jnp.ones((1, 2))
    params = made.init(random.PRNGKey(0), y=y, x=x)
    y = made.apply(params, y=y, x=x)
    a, b = unstack(y, -1)
    chex.assert_trees_all_equal(a, jnp.array(expected))
    chex.assert_trees_all_equal(b, jnp.array(expected))
