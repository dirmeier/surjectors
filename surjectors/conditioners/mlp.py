import haiku as hk
import jax
from jax import numpy as jnp


def mlp_conditioner(
    dims,
    activation=jax.nn.gelu,
    w_init=hk.initializers.TruncatedNormal(stddev=0.01),
    b_init=jnp.zeros,
):
    """
    Create a conditioner network based on an MLP

    Returns
    -------
    hk.Module
        a transformable haiku neural network module
    """

    return hk.nets.MLP(
        output_sizes=dims,
        w_init=w_init,
        b_init=b_init,
        activation=activation,
    )
