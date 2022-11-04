import numpy as np
import jax
from jax import random, numpy as jnp
import matplotlib.pyplot as plt


from surjectors.data import Simulator

simulator = Simulator()

n = 1000
pns = [None] * n
for i in np.arange(n):
    p0, alpha1, alpha2, epsilon_max, f, pn = simulator.sample(
        jnp.array([549229066, 500358972], dtype=jnp.uint32), 100
    )
    pns[i] = pn
