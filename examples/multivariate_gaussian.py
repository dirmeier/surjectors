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
