from typing import Callable, List, Optional, Tuple, Union

import haiku as hk
import jax
from jax import numpy as jnp
from tensorflow_probability.substrates.jax.bijectors.masked_autoregressive import (  # noqa: E501
    _make_dense_autoregressive_masks,
)

from surjectors.conditioners.nn.masked_linear import MaskedLinear


# pylint: disable=too-many-arguments, arguments-renamed
class MADE(hk.Module):
    """
    Masked Autoregressive Density Estimator

    Passing a value through a MADE will output a tensor of shape
    [..., input_size, n_params]
    """

    def __init__(
        self,
        input_size: int,
        hidden_layer_sizes: Union[List[int], Tuple[int]],
        n_params: int,
        w_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
        activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
    ):
        """
        Constructor

        Parameters
        ----------
        input_size: int
            number of input features
        hidden_layer_sizes: Union[List[int], Tuple[int]]
            list/tuple of ints describing the number of nodes in the hidden
            layers
        n_params: int
            number of output parameters. For instance, if used as conditioner
            of a affine bijector, should be 2 (mean and shift)
        w_init: Optional[hk.initializers.Initializer]
            a Haiku initializer
        b_init : Optional[hk.initializers.Initializer]
            a Haiku initializer
        activation: Callable
            an activation function
        """

        super().__init__()
        self.input_size = input_size
        self.output_sizes = hidden_layer_sizes
        self.n_params = n_params
        self.w_init = w_init
        self.b_init = b_init
        self.activation = activation
        masks = _make_dense_autoregressive_masks(
            n_params, self.input_size, self.output_sizes
        )

        layers = []
        for mask in masks:
            layers.append(
                MaskedLinear(
                    mask=mask.astype(jnp.float_),
                    w_init=w_init,
                    b_init=b_init,
                )
            )
        self.layers = tuple(layers)

    def __call__(self, y: jnp.ndarray, x: jnp.ndarray = None) -> jnp.ndarray:
        output = self.layers[0](y)
        if x is not None:
            context = hk.Linear(
                self.output_sizes[0], w_init=self.w_init, b_init=self.b_init
            )(x)
            output += context
        output = self.activation(output)
        for i, layer in enumerate(self.layers[1:]):
            output = layer(output)
            if i < len(self.layers[1:]) - 1:
                output = self.activation(output)
        output = hk.Reshape((self.input_size, self.n_params))(output)
        return output

    def reverse(self):
        """Not implemented"""
        raise NotImplementedError("")
