# type: ignore

import distrax
import haiku as hk
from jax import numpy as jnp
from jax import random

from surjectors import (
  RationalQuadraticSplineMaskedCouplingInferenceFunnel,
  TransformedDistribution,
)
from surjectors._src.conditioners.mlp import make_mlp
from surjectors.util import as_model


def _conditional_fn(n_dim):
  decoder_net = make_mlp([4, 4, n_dim * 2])

  def _fn(z):
    params = decoder_net(z)
    mu, log_scale = jnp.split(params, 2, -1)
    return distrax.Independent(distrax.Normal(mu, jnp.exp(log_scale)))

  return _fn


def _base_distribution_fn(n_latent):
  base_distribution = distrax.Independent(
    distrax.Normal(jnp.zeros(n_latent), jnp.ones(n_latent)),
    reinterpreted_batch_ndims=1,
  )
  return base_distribution


def _get_funnel_surjector(n_latent, n_dimension):
  return RationalQuadraticSplineMaskedCouplingInferenceFunnel(
    n_latent,
    _conditional_fn(n_dimension - n_latent),
    hk.Sequential(
      [
        hk.nets.MLP([4, 4], activate_final=True),
        hk.Linear(n_dimension * 4),
        hk.Reshape((n_dimension, 4)),
      ]
    ),
    -5.0,
    5.0,
  )


def make_surjector(n_dimension, n_latent):
  def _transformation_fn(n_dimension):
    funnel = _get_funnel_surjector(n_latent, n_dimension)
    return funnel

  def _build_td():
    return TransformedDistribution(
      _base_distribution_fn(n_latent), _transformation_fn(n_dimension)
    )

  return as_model(_build_td, methods=("log_prob",))


def test_rq_masked_coupling_inference_funnel():
  n_dimension, n_latent = 4, 2
  y = random.normal(random.PRNGKey(1), shape=(10, n_dimension))

  flow = make_surjector(n_dimension, n_latent)
  params = flow.init(random.PRNGKey(0), y=y)
  _ = flow.apply.log_prob(params, None, y=y)


def test_conditional_rq_masked_coupling_inference_funnel():
  n_dimension, n_latent = 4, 2
  y = random.normal(random.PRNGKey(1), shape=(10, n_dimension))
  x = random.normal(random.PRNGKey(1), shape=(10, 2))

  flow = make_surjector(n_dimension, n_latent)
  params = flow.init(random.PRNGKey(0), y=y, x=x)
  _ = flow.apply.log_prob(params, None, y=y, x=x)
