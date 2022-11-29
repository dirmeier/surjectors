import distrax
import haiku as hk
import jax
import numpy as np
import optax
from jax import numpy as jnp
from jax import random

from surjectors.bijectors.masked_coupling import MaskedCoupling
from surjectors.conditioners.mlp import mlp_conditioner
from surjectors.distributions.transformed_distribution import (
    TransformedDistribution,
)
from surjectors.surjectors.chain import Chain


def _decoder_fn(n_dimension, n_latent):
    decoder_net = mlp_conditioner([32, 32, n_dimension - n_latent])

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


def _get_bijector(n_dimension, n_latent):
    def _transformation_fn():
        layers = []
        mask = jnp.arange(0, np.prod(n_dimension)) % 2
        mask = jnp.reshape(mask, n_dimension)
        mask = mask.astype(bool)

        for _ in range(4):
            layer = MaskedCoupling(
                mask=mask,
                bijector=_bijector_fn,
                conditioner=mlp_conditioner([32, 32, n_dimension]),
            )
            layers.append(layer)
            mask = jnp.logical_not(mask)

        return Chain(layers)

    def _flow(method, **kwargs):
        td = TransformedDistribution(
            _base_distribution_fn(n_dimension), _transformation_fn()
        )
        return td(method, **kwargs)

    td = hk.transform(_flow)
    return td


def _train(
    rng_seq, sampler, surjector_fn, n_data, n_latent, batch_size, n_iter
):
    flow = surjector_fn(n_data, n_latent)

    @jax.jit
    def step(params, state, y_batch, noise_batch, rng):
        def loss_fn(params):
            lp = flow.apply(
                params, rng, method="log_prob", y=y_batch, x=noise_batch
            )
            return -jnp.sum(lp)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_state = adam.update(grads, state, params)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, new_state

    y_init, _, noise_init = sampler(next(rng_seq))
    params = flow.init(next(rng_seq), method="log_prob", y=y_init, x=noise_init)
    adam = optax.adamw(0.001)
    state = adam.init(params)

    losses = [0] * n_iter
    for i in range(n_iter):
        y_batch, _, noise_batch = sampler(next(rng_seq))
        loss, params, state = step(
            params, state, y_batch, noise_batch, next(rng_seq)
        )
        losses[i] = loss

    return params, flow


def _evaluate(rng_seq, params, model, sampler):
    y_batch, _, noise_batch = sampler(next(rng_seq))
    model.apply(
        params, next(rng_seq), method="log_prob", y=y_batch, x=noise_batch
    )


def test_bijector(sampler):
    n_batch, n_data, n_latent = 64, 10, 5
    sampling_fn, _ = sampler(random.PRNGKey(0), n_batch, n_data, n_latent)
    rng_seq = hk.PRNGSequence(0)
    params, flow = _train(
        rng_seq=rng_seq,
        sampler=sampling_fn,
        surjector_fn=_get_bijector,
        n_iter=5,
        batch_size=n_batch,
        n_data=n_data,
        n_latent=n_latent,
    )
    _evaluate(rng_seq, params, flow, sampling_fn)
