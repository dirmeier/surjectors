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
from surjectors.distributions.transformed_distribution import (
    TransformedDistribution,
)
from surjectors.surjectors.affine_masked_coupling_generative_funnel import \
    AffineMaskedCouplingGenerativeFunnel
from surjectors.surjectors.augment import Augment
from surjectors.surjectors.chain import Chain

config.update("jax_enable_x64", True)


def _get_sampler_and_loadings(rng_key, batch_size, n_dimension):
    pz_mean = jnp.array([-2.31, 0.421, 0.1, 3.21, -0.41, -2.31, 0.421, 0.1, 3.21, -0.41])
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
        return y, z, noise
        return z[:, :n_dimension], z , noise

    return _fn, loadings


def _get_slice_surjector(n_dimension, n_latent):
    def _conditioner(dim):
        return hk.Sequential(
            [
                hk.Linear(
                    32,
                    w_init=hk.initializers.TruncatedNormal(stddev=0.01),
                    b_init=jnp.zeros,
                ),
                jax.nn.gelu,
                hk.Linear(
                    32,
                    w_init=hk.initializers.TruncatedNormal(stddev=0.01),
                    b_init=jnp.zeros,
                ),
                jax.nn.gelu,
                hk.Linear(dim * 2),
            ]
        )

    def _encoder_fn():
        decoder_net = _conditioner((n_latent - n_dimension))

        def _fn(z):
            params = decoder_net(z)
            mu, log_scale = jnp.split(params, 2, -1)
            return distrax.Independent(distrax.Normal(mu, jnp.exp(log_scale)))

        return _fn

    def _bijector_fn(params):
        means, log_scales = jnp.split(params, 2, -1)
        return distrax.ScalarAffine(means, jnp.exp(log_scales))

    def _transformation_fn():
        layers = []

        mask = jnp.arange(0, np.prod(n_dimension)) % 2
        mask = jnp.reshape(mask, n_dimension)
        mask = mask.astype(bool)
        for _ in range(2):
            layer = MaskedCoupling(
                mask=mask,
                bijector=_bijector_fn,
                conditioner=_conditioner(n_dimension),
            )
            layers.append(layer)

        layers.append(
            Augment(n_dimension, _encoder_fn())
        )

        mask = jnp.arange(0, np.prod(n_latent)) % 2
        mask = jnp.reshape(mask, n_latent)
        mask = mask.astype(bool)
        for _ in range(2):
            layer = MaskedCoupling(
                mask=mask,
                bijector=_bijector_fn,
                conditioner=_conditioner(n_latent),
            )
            layers.append(layer)
            mask = jnp.logical_not(mask)
        #return Augment(n_dimension, _encoder_fn())
        return Chain(layers)

    def _base_fn():
        base_distribution = distrax.Independent(
            distrax.Normal(jnp.zeros(n_latent), jnp.ones(n_latent)),
            reinterpreted_batch_ndims=1,
        )
        return base_distribution

    def _flow(method, **kwargs):
        td = TransformedDistribution(_base_fn(), _transformation_fn())
        return td(method, **kwargs)

    td = hk.transform(_flow)
    return td


def _get_funnel_surjector(n_dimension, n_latent):
    def _conditioner(dim):
        return hk.Sequential(
            [
                hk.Linear(
                    32,
                    w_init=hk.initializers.TruncatedNormal(stddev=0.01),
                    b_init=jnp.zeros,
                ),
                jax.nn.gelu,
                hk.Linear(
                    32,
                    w_init=hk.initializers.TruncatedNormal(stddev=0.01),
                    b_init=jnp.zeros,
                ),
                jax.nn.gelu,
                hk.Linear(dim * 2),
            ]
        )

    def _encoder_fn():
        decoder_net = _conditioner((n_latent - n_dimension))

        def _fn(z):
            params = decoder_net(z)
            mu, log_scale = jnp.split(params, 2, -1)
            return distrax.Independent(distrax.Normal(mu, jnp.exp(log_scale)))

        return _fn

    def _bijector_fn(params):
        means, log_scales = jnp.split(params, 2, -1)
        return distrax.ScalarAffine(means, jnp.exp(log_scales))

    def _transformation_fn():
        layers = []

        mask = jnp.arange(0, np.prod(n_dimension)) % 2
        mask = jnp.reshape(mask, n_dimension)
        mask = mask.astype(bool)
        for _ in range(2):
            layer = MaskedCoupling(
                mask=mask,
                bijector=_bijector_fn,
                conditioner=_conditioner(n_dimension),
            )
            layers.append(layer)

        layers.append(
            AffineMaskedCouplingGenerativeFunnel(
                n_dimension, _encoder_fn(),  _conditioner(n_latent)
            )
        )

        mask = jnp.arange(0, np.prod(n_latent)) % 2
        mask = jnp.reshape(mask, n_latent)
        mask = mask.astype(bool)
        for _ in range(2):
            layer = MaskedCoupling(
                mask=mask,
                bijector=_bijector_fn,
                conditioner=_conditioner(n_latent),
            )
            layers.append(layer)
            mask = jnp.logical_not(mask)

        return Chain(layers)
        # return AffineMaskedCouplingGenerativeFunnel(
        #     n_dimension, _encoder_fn(),  _conditioner(n_latent)
        # )

    def _base_fn():
        base_distribution = distrax.Independent(
            distrax.Normal(jnp.zeros(n_latent), jnp.ones(n_latent)),
            reinterpreted_batch_ndims=1,
        )
        return base_distribution

    def _flow(method, **kwargs):
        td = TransformedDistribution(_base_fn(), _transformation_fn())
        return td(method, **kwargs)

    td = hk.transform(_flow)
    return td


def train(key, surjector_fn, n_data, n_latent, batch_size, n_iter):
    rng_seq = hk.PRNGSequence(0)
    pyz, loadings = _get_sampler_and_loadings(next(rng_seq), 2*batch_size, n_data)
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

    y_init, _, noise_init = pyz(random.fold_in(next(rng_seq), 0))
    params = flow.init(
        random.PRNGKey(key),
        method="log_prob",
        y=y_init,
        x=noise_init
    )
    adam = optax.adamw(0.001)
    state = adam.init(params)

    losses = [0] * n_iter
    for i in range(n_iter):
        y_batch, _, noise_batch = pyz(next(rng_seq))
        loss, params, state = step(params, state, y_batch, noise_batch, next(rng_seq))
        losses[i] = loss

    losses = jnp.asarray(losses)
    plt.plot(losses)
    plt.show()

    y_batch, z_batch, noise_batch = pyz(next(rng_seq))
    y_pred = flow.apply(
        params, next(rng_seq), method="sample", x=noise_batch,
    )
    print(y_batch[:5, :])
    print(y_pred[:5, :])


def run():
    train(
        key=0,
        surjector_fn=_get_funnel_surjector,
        n_iter=2000,
        batch_size=64,
        n_data=5,
        n_latent=10
    )


if __name__ == "__main__":
    run()
