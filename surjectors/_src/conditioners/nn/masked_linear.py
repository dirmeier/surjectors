from typing import Optional

import chex
import haiku as hk
import numpy as np
from jax import lax
from jax import numpy as jnp


# pylint: disable=too-many-arguments,too-few-public-methods
class MaskedLinear(hk.Module):
    """Linear layer that masks some weights."""

    def __init__(
        self,
        mask: chex.Array,
        with_bias: bool = True,
        w_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
    ):
        """Construct a MaskedLinear layer.

        Args:
            mask: boolean mask
            with_bias: boolean
            w_init: haiku initializer
            b_init: haiku initializer
        """
        super().__init__()
        self.input_size = None
        self.output_size = mask.shape[-1]
        self.with_bias = with_bias
        self.w_init = w_init
        self.b_init = b_init or jnp.zeros
        self.mask = mask

    def __call__(
        self, inputs, *, precision: Optional[lax.Precision] = None
    ) -> jnp.ndarray:
        """Apply the layer."""
        if not inputs.shape:
            raise ValueError("Input must not be scalar.")
        dtype = inputs.dtype

        w_init = self.w_init
        if w_init is None:
            stddev = 1.0 / np.sqrt(jnp.shape(self.mask)[0])
            w_init = hk.initializers.TruncatedNormal(stddev=stddev)
        w = hk.get_parameter("w", jnp.shape(self.mask), dtype, init=w_init)

        outputs = jnp.dot(inputs, jnp.multiply(w, self.mask), precision=None)

        if self.with_bias:
            b = hk.get_parameter(
                "b", (jnp.shape(self.mask)[-1],), dtype, init=self.b_init
            )
            b = jnp.broadcast_to(b, outputs.shape)
            outputs = outputs + b

        return outputs
