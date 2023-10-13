import haiku as hk
import jax
from jax import numpy as jnp


def make_mlp(
    dims,
    activation=jax.nn.gelu,
    w_init=hk.initializers.TruncatedNormal(stddev=0.01),
    b_init=jnp.zeros,
):
    """Create a conditioner network based on an MLP.

    Args:
        dims: dimensions of hidden layers and last layer
        activation: a JAX activation function
        w_init: a haiku initializer
        b_init: a haiku initializer

    Returns:
        a transformable haiku neural network module
    """
    return hk.nets.MLP(
        output_sizes=dims,
        w_init=w_init,
        b_init=b_init,
        activation=activation,
    )
