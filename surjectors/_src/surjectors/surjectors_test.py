# pylint: skip-file

import distrax
import haiku as hk
import jax
import optax
import pytest
from jax import numpy as jnp
from jax import random

from surjectors import (
    AffineMaskedCouplingGenerativeFunnel,
    AffineMaskedCouplingInferenceFunnel,
    Augment,
    Chain,
    MaskedCoupling,
    Slice,
    TransformedDistribution,
)
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


def _conditional_fn(n_dim):
    decoder_net = mlp_conditioner([32, 32, n_dim * 2])

    def _fn(z):
        params = decoder_net(z)
        mu, log_scale = jnp.split(params, 2, -1)
        return distrax.Independent(distrax.Normal(mu, jnp.exp(log_scale)))

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


def _get_augment_surjector(n_latent, n_dimension):
    return Augment(n_dimension, _conditional_fn(n_latent - n_dimension))


def _get_generative_funnel_surjector(n_latent, n_dimension):
    return AffineMaskedCouplingGenerativeFunnel(
        n_dimension,
        _conditional_fn(n_latent - n_dimension),
        mlp_conditioner([32, 32, n_latent * 2]),
    )


def make_surjector(n_dimension, n_latent, surjector_fn):
    def _transformation_fn(n_dimension):
        layers = []
        for i in range(5):
            if i != 3:
                mask = make_alternating_binary_mask(n_dimension, i % 2 == 0)
                layer = MaskedCoupling(
                    mask=mask,
                    bijector=_bijector_fn,
                    conditioner=mlp_conditioner([32, 32, n_dimension * 2]),
                )
            else:
                layer = surjector_fn(n_latent, n_dimension)
                n_dimension = n_latent
            layers.append(layer)
        return Chain(layers)

    def _flow(method, **kwargs):
        td = TransformedDistribution(
            _base_distribution_fn(n_latent), _transformation_fn(n_dimension)
        )
        return td(method, **kwargs)

    td = hk.transform(_flow)
    return td


def _get_slice_surjector(n_latent, n_dimension):
    return Slice(n_latent, _conditional_fn(n_dimension - n_latent))


def _get_inference_funnel_surjector(n_latent, n_dimension):
    return AffineMaskedCouplingInferenceFunnel(
        n_latent,
        _conditional_fn(n_dimension - n_latent),
        mlp_conditioner([32, 32, n_dimension * 2]),
    )


@pytest.fixture(
    params=[
        (_get_generative_funnel_surjector, 5, 10),
        (_get_augment_surjector, 5, 10),
    ],
    ids=["funnel", "augment"],
)
def generative_surjection(request):
    yield request.param


@pytest.fixture(
    params=[
        (_get_inference_funnel_surjector, 10, 5),
        (_get_slice_surjector, 10, 5),
    ],
    ids=["funnel", "slice"],
)
def inference_surjection(request):
    yield request.param


def train(rng_seq, model, sampler, n_iter=5):
    @jax.jit
    def step(rng, params, state, **batch):
        def loss_fn(params):
            lp = model.apply(params, rng, method="log_prob", **batch)
            return -jnp.sum(lp)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_state = adam.update(grads, state, params)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, new_state

    init_data = sampler(next(rng_seq))
    params = model.init(next(rng_seq), method="log_prob", **init_data)

    adam = optax.adamw(0.001)
    state = adam.init(params)

    losses = [0] * n_iter
    for i in range(n_iter):
        batch = sampler(next(rng_seq))
        loss, params, state = step(next(rng_seq), params, state, **batch)
        losses[i] = loss

    return params


def _surjection(surjector_fn, n_data, n_latent):
    rng_seq = hk.PRNGSequence(0)
    sampling_fn = simple_dataset(next(rng_seq), 64, n_data, n_latent)
    model = make_surjector(n_data, n_latent, surjector_fn)
    train(rng_seq, model, sampling_fn)


def test_generative_surjection(generative_surjection):
    _surjection(*generative_surjection)


def test_inference_surjection(inference_surjection):
    _surjection(*inference_surjection)
