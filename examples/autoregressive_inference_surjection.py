from collections import namedtuple

import distrax
import haiku as hk
import jax
import numpy as np
import optax
from jax import numpy as jnp
from jax import random as jr
from matplotlib import pyplot as plt

from surjectors import (
    AffineMaskedAutoregressiveInferenceFunnel,
    Chain,
    MaskedAutoregressive,
    TransformedDistribution,
)
from surjectors.conditioners import MADE, mlp_conditioner
from surjectors.util import as_batch_iterator, unstack


def _decoder_fn(n_dim):
    decoder_net = mlp_conditioner([4, 4, n_dim * 2])

    def _fn(z):
        params = decoder_net(z)
        mu, log_scale = jnp.split(params, 2, -1)
        return distrax.Independent(distrax.Normal(mu, jnp.exp(log_scale)))

    return _fn


def _made_bijector_fn(params):
    means, log_scales = unstack(params, -1)
    return distrax.Inverse(distrax.ScalarAffine(means, jnp.exp(log_scales)))


def make_model(n_dimensions):
    def _flow(**kwargs):
        n_dim = n_dimensions
        layers = []
        for i in range(3):
            if i != 1:
                layer = AffineMaskedAutoregressiveInferenceFunnel(
                    n_keep=int(n_dim / 2),
                    decoder=_decoder_fn(int(n_dim / 2)),
                    conditioner=MADE(int(n_dim / 2), [8, 8], 2),
                )
                n_dim = int(n_dim / 2)
            else:
                layer = MaskedAutoregressive(
                    conditioner=MADE(n_dim, [8, 8], 2),
                    bijector_fn=_made_bijector_fn,
                )
            layers.append(layer)
        chain = Chain(layers)

        base_distribution = distrax.Independent(
            distrax.Normal(jnp.zeros(n_dim), jnp.ones(n_dim)),
            reinterpreted_batch_ndims=1,
        )
        td = TransformedDistribution(base_distribution, chain)
        return td.log_prob(**kwargs)

    td = hk.transform(_flow)
    td = hk.without_apply_rng(td)
    return td


def train(rng_seq, data, model, max_n_iter=1000):
    train_iter = as_batch_iterator(next(rng_seq), data, 100, True)
    params = model.init(next(rng_seq), **train_iter(0))

    optimizer = optax.adam(1e-4)
    state = optimizer.init(params)

    @jax.jit
    def step(params, state, **batch):
        def loss_fn(params):
            lp = model.apply(params, **batch)
            return -jnp.sum(lp)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_state = optimizer.update(grads, state, params)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, new_state

    losses = np.zeros(max_n_iter)
    for i in range(max_n_iter):
        train_loss = 0.0
        for j in range(train_iter.num_batches):
            batch = train_iter(j)
            batch_loss, params, state = step(params, state, **batch)
            train_loss += batch_loss
        losses[i] = train_loss

    return params, losses


def run():
    n, p = 1000, 20
    rng_seq = hk.PRNGSequence(2)
    y = jr.normal(next(rng_seq), shape=(n, p))
    data = namedtuple("named_dataset", "y")(y)

    model = make_model(p)
    params, losses = train(rng_seq, data, model)
    plt.plot(losses)
    plt.show()

    y = jr.normal(next(rng_seq), shape=(10, p))
    print(model.apply(params, **{"y": y}))


if __name__ == "__main__":
    run()
