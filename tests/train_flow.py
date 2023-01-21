import jax
import optax
from jax import numpy as jnp


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
