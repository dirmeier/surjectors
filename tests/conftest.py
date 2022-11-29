import distrax
import pytest
from jax import numpy as jnp
from jax import random


@pytest.fixture
def sampler():
    def _sampler(rng_key, batch_size, n_dimension, n_latent):

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
            return y, z, noise

        return _fn, loadings

    return _sampler
