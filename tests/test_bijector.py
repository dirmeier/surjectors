import chex
import distrax
import haiku as hk
import jax
import optax
import pytest
import sampler
from jax import numpy as jnp
from jax import random

import surjectors
from surjectors.conditioners.mlp import mlp_conditioner
from surjectors.util import make_alternating_binary_mask


def _bijector_fn(params):
    means, log_scales = jnp.split(params, 2, -1)
    return distrax.ScalarAffine(means, jnp.exp(log_scales))


def _base_distribution_fn(n_latent):
    base_distribution = distrax.Independent(
        distrax.Normal(jnp.zeros(n_latent), jnp.ones(n_latent)),
        reinterpreted_batch_ndims=1,
    )
    return base_distribution


def masked_coupling_bijector(n_dim, td_ctor, flow_ctor):
    def _transformation_fn(n_dimension):
        mask = make_alternating_binary_mask(n_dimension, 0 % 2 == 0)
        layer = flow_ctor(
            mask=mask,
            bijector=_bijector_fn,
            conditioner=mlp_conditioner(
                [8, n_dim * 2],
                w_init=hk.initializers.TruncatedNormal(stddev=1.0),
                b_init=jnp.ones,
            ),
        )

        return layer

    def _flow(y):
        td = td_ctor(_base_distribution_fn(n_dim), _transformation_fn(n_dim))
        return td.log_prob(y)

    td = hk.transform(_flow)
    td = hk.without_apply_rng(td)
    return td


@pytest.fixture(
    params=[
        (
            masked_coupling_bijector,
            (distrax.Transformed, distrax.MaskedCoupling),
            (surjectors.TransformedDistribution, surjectors.MaskedCoupling),
        ),
    ],
    ids=["masked_coupling"],
)
def bijection(request):
    yield request.param


def test_params_against_distrax_bijector(bijection):
    rng_seq = hk.PRNGSequence(0)
    n_dim, n_dim_latent = 2, 2

    bijector_fn, distrax_ctors, surjectors_ctors = bijection
    distrax_model = bijector_fn(n_dim, *distrax_ctors)
    surjectors_model = bijector_fn(n_dim, *surjectors_ctors)

    sampling_fn, _ = sampler.sampler(next(rng_seq), 64, n_dim, n_dim_latent)
    init_data = sampling_fn(next(rng_seq))

    rng = next(rng_seq)
    params_distrax = distrax_model.init(rng, init_data["y"])
    params_surjectors = surjectors_model.init(rng, y=init_data["y"])

    chex.assert_trees_all_equal(params_distrax, params_surjectors)
    jnp.array_equal(
        distrax_model.apply(params_distrax, init_data["y"]),
        surjectors_model.apply(params_surjectors, y=init_data["y"]),
    )
    jnp.array_equal(
        distrax_model.apply(params_surjectors, init_data["y"]),
        surjectors_model.apply(params_distrax, y=init_data["y"]),
    )


def train(rng_key, params, model, sampler, n_iter=5):
    @jax.jit
    def step(params, state, y):
        def loss_fn(params):
            lp = model.apply(params, y)
            return -jnp.sum(lp)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_state = adam.update(grads, state, params)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, new_state

    adam = optax.adamw(0.001)
    state = adam.init(params)

    for i in range(n_iter):
        rng = random.fold_in(rng_key, i)
        batch = sampler(rng)
        _, params, state = step(params, state, batch["y"])

    return params


def test_against_distrax_bijector_after_training(bijection):
    rng_seq = hk.PRNGSequence(0)
    n_dim, n_dim_latent = 2, 2

    bijector_fn, distrax_ctors, surjectors_ctors = bijection
    distrax_model = bijector_fn(n_dim, *distrax_ctors)
    surjectors_model = bijector_fn(n_dim, *surjectors_ctors)

    sampling_fn, _ = sampler.sampler(next(rng_seq), 64, n_dim, n_dim_latent)
    init_data = sampling_fn(next(rng_seq))

    init_rng = next(rng_seq)
    train_rng = next(rng_seq)
    params_distrax = distrax_model.init(init_rng, init_data["y"])
    params_distrax = train(
        train_rng, params_distrax, distrax_model, sampling_fn
    )

    params_surjectors = surjectors_model.init(init_rng, y=init_data["y"])
    params_surjectors = train(
        train_rng, params_surjectors, surjectors_model, sampling_fn
    )

    chex.assert_trees_all_equal(params_distrax, params_surjectors)
