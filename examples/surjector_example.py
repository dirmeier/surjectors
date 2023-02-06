"""
Bayesian Neural Network
=======================

This example implements the training and prediction of a
Bayesian Neural Network.
Predictions from a Haiku MLP fro the same data are shown
as a reference.
References
----------
[1] Blundell C., Cornebise J., Kavukcuoglu K., Wierstra D.
    "Weight Uncertainty in Neural Networks".
    ICML, 2015.
"""


import distrax
import haiku as hk
import jax
import numpy as np
import optax
from jax import numpy as jnp
from jax import random
from matplotlib import pyplot as plt

from surjectors import (
    Chain,
    MaskedCoupling,
    TransformedDistribution,
    mlp_conditioner,
)
from surjectors.util import (
    as_batch_iterator,
    make_alternating_binary_mask,
    named_dataset,
)


def simulator_fn(seed, theta):
    p_noise = distrax.Normal(jnp.zeros_like(theta), 1.0)
    noise = p_noise.sample(seed=seed)
    return theta + 0.1 * noise


def make_model(dim):
    def _bijector_fn(params):
        means, log_scales = jnp.split(params, 2, -1)
        return distrax.ScalarAffine(means, jnp.exp(log_scales))

    def _flow(**kwargs):
        layers = []
        for i in range(2):
            mask = make_alternating_binary_mask(dim, i % 2 == 0)
            layer = MaskedCoupling(
                mask=mask,
                bijector=_bijector_fn,
                conditioner=mlp_conditioner([8, 8, dim * 2]),
            )
            layers.append(layer)
        chain = Chain(layers)

        base_distribution = distrax.Independent(
            distrax.Normal(jnp.zeros(dim), jnp.ones(dim)),
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
    n = 1000
    prior = distrax.Uniform(jnp.full(2, -2), jnp.full(2, 2))
    theta = prior.sample(seed=random.PRNGKey(0), sample_shape=(n,))
    likelihood = distrax.MultivariateNormalDiag(theta, jnp.ones_like(theta))
    y = likelihood.sample(seed=random.PRNGKey(1))
    data = named_dataset(y, theta)

    model = make_model(2)
    params, losses = train(hk.PRNGSequence(2), data, model)
    plt.plot(losses)
    plt.show()

    theta = jnp.ones((5, 2))
    data = jnp.repeat(jnp.arange(5), 2).reshape(-1, 2)
    print(model.apply(params, **{"y": data, "x": theta}))


if __name__ == "__main__":
    run()
