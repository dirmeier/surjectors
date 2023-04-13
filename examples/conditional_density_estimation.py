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
    MaskedAutoregressive,
    MaskedCoupling,
    Permutation,
    TransformedDistribution,
)
from surjectors.conditioners import MADE, mlp_conditioner
from surjectors.util import (
    as_batch_iterator,
    make_alternating_binary_mask,
    named_dataset,
    unstack,
)


def make_model(dim, model="coupling"):
    def _bijector_fn(params):
        means, log_scales = unstack(params, -1)
        return distrax.ScalarAffine(means, jnp.exp(log_scales))

    def _flow(method, **kwargs):
        layers = []
        order = jnp.arange(2)
        for i in range(2):
            if model == "coupling":
                mask = make_alternating_binary_mask(2, i % 2 == 0)
                layer = MaskedCoupling(
                    mask=mask,
                    bijector=_bijector_fn,
                    conditioner=hk.Sequential(
                        [
                            mlp_conditioner([8, 8, dim * 2]),
                            hk.Reshape((dim, dim)),
                        ]
                    ),
                )
                layers.append(layer)
            else:
                layer = MaskedAutoregressive(
                    bijector_fn=_bijector_fn,
                    conditioner=MADE(
                        2,
                        [32, 32, 2 * 2],
                        2,
                        w_init=hk.initializers.TruncatedNormal(0.01),
                        b_init=jnp.zeros,
                    ),
                )
                order = order[::-1]
                layers.append(layer)
                layers.append(Permutation(order, 1))
        if model != "coupling":
            layers = layers[:-1]
        chain = Chain(layers)

        base_distribution = distrax.Independent(
            distrax.Normal(jnp.zeros(dim), jnp.ones(dim)),
            reinterpreted_batch_ndims=1,
        )
        td = TransformedDistribution(base_distribution, chain)
        return td(method=method, **kwargs)

    td = hk.transform(_flow)
    return td


def train(rng_seq, data, model, max_n_iter=1000):
    train_iter = as_batch_iterator(next(rng_seq), data, 100, True)
    params = model.init(next(rng_seq), method="log_prob", **train_iter(0))

    optimizer = optax.adam(1e-4)
    state = optimizer.init(params)

    @jax.jit
    def step(params, state, **batch):
        def loss_fn(params):
            lp = model.apply(params, None, method="log_prob", **batch)
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
    n = 10000
    thetas = distrax.Normal(jnp.zeros(2), jnp.full(2, 10)).sample(
        seed=random.PRNGKey(0), sample_shape=(n,)
    )
    y = 2 * thetas + distrax.Normal(jnp.zeros_like(thetas), 0.1).sample(
        seed=random.PRNGKey(1)
    )
    data = named_dataset(y, thetas)

    model = make_model(2)
    params, losses = train(hk.PRNGSequence(2), data, model)
    samples = model.apply(
        params,
        random.PRNGKey(2),
        method="sample",
        x=jnp.full_like(thetas, -2.0),
    )

    plt.hist(samples[:, 0])
    plt.hist(samples[:, 1])
    plt.show()


if __name__ == "__main__":
    run()
