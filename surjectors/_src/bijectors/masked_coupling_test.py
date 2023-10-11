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
from surjectors.conditioners.mlp import mlp_conditioner
from surjectors.util import make_alternating_binary_mask


def simple_dataset(rng_key, batch_size, n_dimension, n_latent):
    means_sample_key, rng_key = random.split(rng_key, 2)
    pz_mean = distrax.Normal(0.0, 10.0).sample(
        seed=means_sample_key, sample_shape=(n_latent)
    )
    pz = distrax.MultivariateNormalDiag(
        loc=pz_mean, scale_diag=jnp.ones_like(pz_mean)
    )
    p_loadings = distrax.Normal(0.0, 10.0)
    make_noise = distrax.Normal(0.0, 1)

    loadings_sample_key, rng_key = random.split(rng_key, 2)
    loadings = p_loadings.sample(
        seed=loadings_sample_key, sample_shape=(n_dimension, len(pz_mean))
    )

    def _fn(rng_key):
        z_sample_key, noise_sample_key = random.split(rng_key, 2)
        z = pz.sample(seed=z_sample_key, sample_shape=(batch_size,))
        noise = make_noise.sample(
            seed=noise_sample_key, sample_shape=(batch_size, n_dimension)
        )

        y = (loadings @ z.T).T + noise
        return {"y": y, "x": noise}

    return _fn


def slcp(rng_key, batch_size):
    prior = distrax.Uniform(jnp.full(2, -2), jnp.full(2, 2))

    def _fn(rng_key):
        y_sample_key, noise_sample_key = random.split(rng_key, 2)
        theta = prior.sample(seed=noise_sample_key, sample_shape=(batch_size,))
        likelihood = distrax.MultivariateNormalDiag(theta, jnp.ones_like(theta))
        y = likelihood.sample(seed=y_sample_key)

        return {"y": y, "x": theta}

    return _fn


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


def masked_conditional_coupling_bijector(n_dim):
    def _flow(**kwargs):
        layers = []
        for i in range(2):
            mask = make_alternating_binary_mask(n_dim, i % 2 == 0)
            layer = surjectors.MaskedCoupling(
                mask=mask,
                bijector=_bijector_fn,
                conditioner=mlp_conditioner([8, 8, n_dim * 2]),
            )
            layers.append(layer)
        chain = surjectors.Chain(layers)

        base_distribution = distrax.Independent(
            distrax.Normal(jnp.zeros(n_dim), jnp.ones(n_dim)),
            reinterpreted_batch_ndims=1,
        )
        td = surjectors.TransformedDistribution(base_distribution, chain)
        return td.log_prob(**kwargs)

    td = hk.transform(_flow)
    td = hk.without_apply_rng(td)
    return td


def _train_unconditional(rng_key, params, model, sampler, n_iter=5):
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


def _train_conditional(rng_key, params, model, sampler, n_iter=1000):
    @jax.jit
    def step(params, state, **batch):
        def loss_fn(params):
            lp = model.apply(params, **batch)
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
        _, params, state = step(params, state, **batch)

    return params


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

    sampling_fn = simple_dataset(next(rng_seq), 64, n_dim, n_dim_latent)
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


def test_against_distrax_bijector_after_training(bijection):
    rng_seq = hk.PRNGSequence(0)
    n_dim, n_dim_latent = 2, 2

    bijector_fn, distrax_ctors, surjectors_ctors = bijection
    distrax_model = bijector_fn(n_dim, *distrax_ctors)
    surjectors_model = bijector_fn(n_dim, *surjectors_ctors)

    sampling_fn = simple_dataset(next(rng_seq), 64, n_dim, n_dim_latent)
    init_data = sampling_fn(next(rng_seq))

    init_rng = next(rng_seq)
    train_rng = next(rng_seq)
    params_distrax = distrax_model.init(init_rng, init_data["y"])
    params_distrax = _train_unconditional(
        train_rng, params_distrax, distrax_model, sampling_fn
    )

    params_surjectors = surjectors_model.init(init_rng, y=init_data["y"])
    params_surjectors = _train_unconditional(
        train_rng, params_surjectors, surjectors_model, sampling_fn
    )

    chex.assert_trees_all_equal(params_distrax, params_surjectors)


def test_conditional_masked_bijector():
    rng_seq = hk.PRNGSequence(0)
    n_dim = 2
    model = masked_conditional_coupling_bijector(n_dim)

    sampling_fn = slcp(next(rng_seq), 64)
    init_data = sampling_fn(next(rng_seq))

    params = model.init(next(rng_seq), **init_data)
    params = _train_conditional(next(rng_seq), params, model, sampling_fn)

    theta = jnp.ones((5, 2))
    data = jnp.repeat(jnp.arange(5), 2).reshape(-1, 2)
    out = model.apply(params, **{"y": data, "x": theta})
    max_lp_idx = jnp.argmax(out)
    chex.assert_equal(max_lp_idx, 1)
