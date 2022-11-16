import haiku as hk
import jax
from jax import numpy as jnp


def mlp_conditioner(
        dims,
        activation=jax.nn.gelu,
        w_init=hk.initializers.TruncatedNormal(stddev=0.01),
        b_init=jnp.zeros
):
    dims[-1] = dims[-1] * 2

    return hk.nets.MLP(
        output_sizes=dims,
        w_init=w_init,
        b_init=b_init,
        activation=activation
    )
