from collections import namedtuple

import chex
import numpy as np
from jax import lax
from jax import numpy as jnp
from jax import random

named_dataset = namedtuple("named_dataset", "y x")


class _DataLoader:
    def __init__(self, num_batches, idxs, get_batch):
        self.num_batches = num_batches
        self.idxs = idxs
        self.get_batch = get_batch

    def __call__(self, idx, idxs=None):
        if idxs is None:
            idxs = self.idxs
        return self.get_batch(idx, idxs)


def make_alternating_binary_mask(dim, is_even):
    mask = jnp.arange(0, np.prod(dim)) % 2
    mask = jnp.reshape(mask, dim)
    mask = mask.astype(bool)
    if not is_even:
        mask = jnp.logical_not(mask)
    return mask


def as_batch_iterator(
    rng_key: chex.PRNGKey, data: named_dataset, batch_size, shuffle
):
    n = data.y.shape[0]
    if n < batch_size:
        num_batches = 1
        batch_size = n
    elif n % batch_size == 0:
        num_batches = int(n // batch_size)
    else:
        num_batches = int(n // batch_size) + 1

    idxs = jnp.arange(n)
    if shuffle:
        idxs = random.permutation(rng_key, idxs)

    def get_batch(idx, idxs=idxs):
        start_idx = idx * batch_size
        step_size = jnp.minimum(n - start_idx, batch_size)
        ret_idx = lax.dynamic_slice_in_dim(idxs, idx * batch_size, step_size)
        batch = {
            name: lax.index_take(array, (ret_idx,), axes=(0,))
            for name, array in zip(data._fields, data)
        }
        return batch

    return _DataLoader(num_batches, idxs, get_batch)
