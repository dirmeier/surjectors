from typing import Optional

import chex
import haiku as hk
import numpy as np
from jax import lax
from jax import numpy as jnp


# pylint: disable=too-many-arguments,too-few-public-methods
class MaskedLinear(hk.Linear):
    """
    Linear layer that masks some weights out
    """

    def __init__(
        self,
        mask: chex.Array,
        with_bias: bool = True,
        w_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
        name: Optional[str] = None,
    ):

        super().__init__(mask.shape[-1], with_bias, w_init, b_init, name)
        self.mask = mask

    def __call__(
        self, inputs, *, precision: Optional[lax.Precision] = None
    ) -> jnp.ndarray:
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
