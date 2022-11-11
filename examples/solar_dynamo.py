import numpy as np
import jax
from jax import random, numpy as jnp
import matplotlib.pyplot as plt

from examples.solar_dynamo_data import SolarDynamoSimulator

simulator = SolarDynamoSimulator()

n_iter = 1000
for i in np.arange(n_iter):
    p0, alpha1, alpha2, epsilon_max, f, pn = simulator.sample(
        jnp.array([549229066, 500358972], dtype=jnp.uint32), 100
    )
    pns[i] = pn
