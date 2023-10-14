from collections import namedtuple

import numpy as np
from jax import lax
from jax import numpy as jnp
from jax import random as jr

__all__ = ["make_alternating_binary_mask", "as_batch_iterator", "unstack"]

named_dataset = namedtuple("named_dataset", "y x")


# pylint: disable=too-few-public-methods
class _DataLoader:
    """Dataloader class."""

    def __init__(self, num_batches, idxs, get_batch):
        self.num_batches = num_batches
        self.idxs = idxs
        self.get_batch = get_batch

    def __call__(self, idx, idxs=None):
        if idxs is None:
            idxs = self.idxs
        return self.get_batch(idx, idxs)


def make_alternating_binary_mask(n_dim: int, even_idx_as_true: bool = False):
    """Create a binary masked array.

    Args:
        n_dim: length of the masked array to be created
        even_idx_as_true: a boolean indicating which indices are set to zero.
            If even_idx_as_true=True sets all even indices [0, 2, 4, ...]
            to True

    Returns:
        boolean masked array where every even or uneven index is True
    """
    mask = jnp.arange(0, np.prod(n_dim)) % 2
    mask = jnp.reshape(mask, n_dim)
    mask = mask.astype(bool)
    if even_idx_as_true:
        mask = jnp.logical_not(mask)
    return mask


def as_batch_iterator(
    rng_key: jr.PRNGKey, data: named_dataset, batch_size: int, shuffle=True
):
    """Create a batch iterator for a data set.

    Args:
        rng_key: a JAX random key
        data: a data set for which an iterator is created. The data set needs
            to be a NamedTuple with at least one element being called `y`. If a
            conditional flow is to be trained, the second element has to be
            called `x`.
        batch_size: size of each batch of data that is returned by the iterator
        shuffle: if true shuffles the data before creating batches

    Examples:
        >>> from collections import namedtuple
        >>> from jax import numpy as jnp, random as jr
        >>>
        >>> y = jr.normal(jr.PRNGKey(0), (1000, 2))
        >>> as_batch_iterator(jr.PRNGKey(1), namedtuple("data", "y")(y), 100)
        >>>
        >>> x = jr.normal(jr.PRNGKey(1), (1000, 2))
        >>> as_batch_iterator(
        >>>     jr.PRNGKey(1), namedtuple("data", "y x")(y, x), 100
        >>> )

    Returns:
        a data loader object
    """
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
        idxs = jr.permutation(rng_key, idxs)

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


def unstack(x, axis=0):
    """Unstack a tensor.

    Unstack a tensor as tf.unstack does

    Args:
        x: array to unstack
        axis: the axis as integer index

    Returns:
        unstacked array
    """
    return [
        lax.index_in_dim(x, i, axis, keepdims=False)
        for i in range(x.shape[axis])
    ]
