# pylint: skip-file

import chex
import distrax
import haiku as hk
import pytest
from jax import numpy as jnp
from jax import random

from surjectors import TransformedDistribution
from surjectors.bijectors.masked_autoregressive import MaskedAutoregressive
from surjectors.conditioners import MADE
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


@pytest.fixture()
def forward_affine_masked_autoregressive_flow(request):
    def _affine_bijector_fn(params):
        means, log_scales = unstack(params, -1)
        return distrax.ScalarAffine(means, jnp.exp(log_scales))

    input_shape, hidden_shapes, output_shape = request.param

    @hk.without_apply_rng
    @hk.transform
    def _flow(**kwargs):
        layer = MaskedAutoregressive(
            bijector_fn=_affine_bijector_fn,
            conditioner=MADE(
                input_shape,
                hidden_shapes,
                output_shape,
                w_init=jnp.ones,
                b_init=jnp.zeros,
                activation=lambda x: x,
            ),
        )

        return layer.forward(**kwargs)

    return _flow


@pytest.fixture(
    params=[(_rq_bijector_fn, 4), (_affine_bijector_fn, 2)],
    ids=["rq_masked_autoregressive", "affine_masked_autoregressive"],
)
def bijection(request):
    yield request.param


def test_unconditional_bijector_shape(bijection):
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


@pytest.mark.parametrize(
    "forward_affine_masked_autoregressive_flow,expected",
    [
        ((3, [2, 4, 2], 2), [[1.0000000e00, 9.3890562e00, 4.2617353e11]]),
        ((3, [2], 2), [[1.0, 3.7182817, 310.09976]]),
    ],
    indirect=["forward_affine_masked_autoregressive_flow"],
)
def test_unconditional_affine_masked_autoregressive_flow(
    forward_affine_masked_autoregressive_flow, expected
):
    """
    Tested using:

    import numpy as np
    z = np.ones((1, 3), dtype=np.float32)
    ma = AutoregressiveNetwork(
        2,
        3,
        hidden_units=[2, 4, 2], # or [2]
        conditional_input_layers="first_layer",
        kernel_initializer="ones",
        activation=lambda x: x
    )
    flow = MaskedAutoregressiveFlow(ma)
    y = flow.forward(z)
    print(y)
    """
    z = jnp.ones((1, 3))
    params = forward_affine_masked_autoregressive_flow.init(
        random.PRNGKey(0), z=z
    )
    y = forward_affine_masked_autoregressive_flow.apply(params, z=z)
    assert jnp.allclose(y, jnp.array(expected))


@pytest.mark.parametrize(
    "forward_affine_masked_autoregressive_flow,expected",
    [
        ((3, [2, 4, 2], 2), [[1.0000000e00, 1.6063736e01, 2.9474317e18]]),
        ((3, [2], 2), [[1.0000000e00, 4.9692965e00, 1.9453466e03]]),
    ],
    indirect=["forward_affine_masked_autoregressive_flow"],
)
def test_conditional_affine_masked_autoregressive_flow(
    forward_affine_masked_autoregressive_flow, expected
):
    """
    Tested using:

    import numpy as np
    z = np.ones((1, 3), dtype=np.float32)
    x = np.array([[0.1, 0.2]], dtype=np.float32)
    ma = AutoregressiveNetwork(
        2,
        3,
        hidden_units=[2, 4, 2], # or [2]
        conditional_input_layers="first_layer",
        kernel_initializer="ones",
        activation=lambda x: x
    )
    flow = MaskedAutoregressiveFlow(ma)
    y = flow.forward(z, conditional_input=x)
    print(y)
    """

    z = jnp.ones((1, 3))
    x = jnp.array([[0.1, 0.2]])
    params = forward_affine_masked_autoregressive_flow.init(
        random.PRNGKey(0), z=z, x=x
    )
    y = forward_affine_masked_autoregressive_flow.apply(params, z=z, x=x)
    assert jnp.allclose(y, jnp.array(expected))


def test_conditional_bijector_shape(bijection):
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
