import distrax
import haiku as hk
import pytest
import sampler
import train_flow
from jax import numpy as jnp

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


def _surjection(surjector_fn, n_data, n_latent):
    rng_seq = hk.PRNGSequence(0)
    sampling_fn, _ = sampler.sampler(next(rng_seq), 64, n_data, n_latent)
    model = make_surjector(n_data, n_latent, surjector_fn)
    train_flow.train(rng_seq, model, sampling_fn)


def test_generative_surjection(generative_surjection):
    _surjection(*generative_surjection)


def test_inference_surjection(inference_surjection):
    _surjection(*inference_surjection)
