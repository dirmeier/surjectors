import optax
from distrax import LowerUpperTriangularAffine
from jax import numpy as jnp
import jax
import haiku as hk
import distrax

from surjectors.surjectors.funnel import Funnel
from surjectors.surjectors.lu_linear import LULinear


class MLP(Funnel, hk.Module):
    def __init__(self, n_keep, decoder, dtype=jnp.float32):
        self._r = LULinear(n_keep, dtype)
        self._w_prime = hk.Linear(n_keep, with_bias=True)

        self._decoder = decoder # TODO: should be a conditional gaussian
        super().__init__(n_keep, decoder)

    def inverse_and_likelihood_contribution(self, y):
        x_plus, x_minus = y[:, :self.n_keep],  y[:, self.n_keep:]
        z, lc = self._r.inverse_and_likelihood_contribution(x_plus) + self._w_prime(x_minus)
        lp = self._decoder.log_prob(x_minus, context=z)
        return z, lp + lc

    def forward_and_likelihood_contribution(self, z):
        pass

prng = hk.PRNGSequence(jax.random.PRNGKey(42))
matrix = jax.random.uniform(next(prng), (4, 4))
bias = jax.random.normal(next(prng),  (4,))
bijector = LowerUpperTriangularAffine(matrix, bias)

#
# def loss():
#     x, lc = bijector.inverse_and_log_det(jnp.zeros(4) * 2.1)
#     lp = distrax.Normal(jnp.zeros(4)).log_prob(x)
#     return -jnp.sum(lp - lc)
#
# print(bijector.matrix)
#
# adam = optax.adam(0.003)
# g = jax.grad(loss)()
#
# print(g)
#

matrix = jax.random.uniform(next(prng), (4, 4))
bias = jax.random.normal(next(prng),  (4,))
bijector = LowerUpperTriangularAffine(matrix, bias)

n = jnp.ones((4, 4)) * 3.1
n += jnp.triu(n) * 2

bijector = LowerUpperTriangularAffine(n, jnp.zeros(4))


print(bijector.forward(jnp.ones(4)))

print(n @jnp.ones(4) )