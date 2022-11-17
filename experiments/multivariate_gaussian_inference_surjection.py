import distrax
import haiku as hk
import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax import config
from jax import numpy as jnp
from jax import random

from surjectors.bijectors.masked_coupling import MaskedCoupling
from surjectors.conditioners.mlp import mlp_conditioner
from surjectors.distributions.transformed_distribution import (
    TransformedDistribution,
)
from surjectors.surjectors.affine_masked_coupling_inference_funnel import \
    AffineMaskedCouplingInferenceFunnel
from surjectors.surjectors.chain import Chain
from surjectors.surjectors.slice import Slice

config.update("jax_enable_x64", True)


def _get_sampler(rng_key, batch_size, n_dimension, n_latent):
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
        # y = jnp.concatenate([z, z] ,axis=-1)
        return y, z, noise

    return _fn, loadings


def _decoder_fn(n_dimension,  n_latent):
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


def _get_slice_surjector(n_dimension, n_latent):
    def _transformation_fn():
        layers = []
        mask = jnp.arange(0, np.prod(n_dimension)) % 2
        mask = jnp.reshape(mask, n_dimension)
        mask = mask.astype(bool)

        for _ in range(2):
            layer = MaskedCoupling(
                mask=mask,
                bijector=_bijector_fn,
                conditioner=mlp_conditioner([32, 32, n_dimension])
            )
            layers.append(layer)

        layers.append(
            Slice(n_latent, _decoder_fn(n_dimension,  n_latent))
        )

        mask = jnp.arange(0, np.prod(n_latent)) % 2
        mask = jnp.reshape(mask, n_latent)
        mask = mask.astype(bool)
        for _ in range(2):
            layer = MaskedCoupling(
                mask=mask,
                bijector=_bijector_fn,
                conditioner=mlp_conditioner([32, 32, n_latent]),
            )
            layers.append(layer)
            mask = jnp.logical_not(mask)

        return Chain(layers)

    def _flow(method, **kwargs):
        td = TransformedDistribution(_base_distribution_fn(n_latent), _transformation_fn())
        return td(method, **kwargs)

    td = hk.transform(_flow)
    return td


def _get_funnel_surjector(n_dimension, n_latent):
    def _transformation_fn():
        layers = []
        mask = jnp.arange(0, np.prod(n_dimension)) % 2
        mask = jnp.reshape(mask, n_dimension)
        mask = mask.astype(bool)

        for _ in range(2):
            layer = MaskedCoupling(
                mask=mask,
                bijector=_bijector_fn,
                conditioner=mlp_conditioner([32, 32, n_dimension]),
            )
            layers.append(layer)
            mask = jnp.logical_not(mask)

        layers.append(
            AffineMaskedCouplingInferenceFunnel(
                n_latent,
                _decoder_fn(n_dimension, n_latent),
                mlp_conditioner([32, 32, n_dimension])
            )
        )

        mask = jnp.arange(0, np.prod(n_latent)) % 2
        mask = jnp.reshape(mask, n_latent)
        mask = mask.astype(bool)
        for _ in range(2):
            layer = MaskedCoupling(
                mask=mask,
                bijector=_bijector_fn,
                conditioner=mlp_conditioner([32, 32, n_latent]),
            )
            layers.append(layer)
            mask = jnp.logical_not(mask)

        return Chain(layers)

    def _flow(method, **kwargs):
        td = TransformedDistribution(_base_distribution_fn(n_latent), _transformation_fn())
        return td(method, **kwargs)

    td = hk.transform(_flow)
    return td


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
        td = TransformedDistribution(_base_distribution_fn(n_dimension), _transformation_fn())
        return td(method, **kwargs)

    td = hk.transform(_flow)
    return td


def train(rng_seq, sampler, surjector_fn, n_data, n_latent, batch_size, n_iter):
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
    params = flow.init(
        next(rng_seq),
        method="log_prob",
        y=y_init,
        x=noise_init
    )
    adam = optax.adamw(0.001)
    state = adam.init(params)

    losses = [0] * n_iter
    for i in range(n_iter):
        y_batch, _, noise_batch = sampler(next(rng_seq))
        loss, params, state = step(params, state, y_batch, noise_batch, next(rng_seq))
        losses[i] = loss

    losses = jnp.asarray(losses)
    plt.plot(losses)
    plt.show()

    return flow, params


def evaluate(rng_seq, params, model, sampler, batch_size, n_data):
    y_batch, _, noise_batch = sampler(next(rng_seq))
    lp = model.apply(params, next(rng_seq), method="log_prob", y=y_batch, x=noise_batch)
    print("\tPPLP: {:.3f}".format(jnp.mean(lp)))


def run():
    n_iter = 2000
    batch_size = 64
    n_data, n_latent = 100, 75
    sampler, _ = _get_sampler(random.PRNGKey(0), batch_size, n_data, n_latent)
    for method, _fn in [
        ["Slice", _get_slice_surjector],
        ["Funnel", _get_funnel_surjector],
        ["Bijector", _get_bijector]
    ]:
        print(f"Doing {method}")
        rng_seq = hk.PRNGSequence(0)
        model, params = train(
            rng_seq=rng_seq,
            sampler=sampler,
            surjector_fn=_fn,
            n_iter=n_iter,
            batch_size=batch_size,
            n_data=n_data,
            n_latent=n_latent
        )
        evaluate(rng_seq, params, model, sampler, batch_size, n_data)


if __name__ == "__main__":
    run()