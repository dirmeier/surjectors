# An introduction to Surjectors

This introductory notebook illustrates the usage of the basic functionality of `surjectors` such as

- using [Haiku](https://github.com/google-deepmind/dm-haiku), [Distrax](https://github.com/google-deepmind/distrax) and [Optax](https://github.com/google-deepmind/optax),
- defining transformed distributions,
- and training a normalizing flow.

At its core, `surjectors` uses Haiku to compose functions with trainable parameters, such as neural networks, and Distrax to compute probability densities and sample random variables. Hence using `surjectors` also requires some knowledge of these two.
Below we illustrate how to use Haiku's core functionality to build pure functions and how to build normalizing flows for density estimation. You can find an interactive version here:

```python
from collections import namedtuple

import distrax
import haiku as hk
import jax
import numpy as np
import optax
from jax import jit
from jax import numpy as jnp
from jax import random as jr

import matplotlib.pyplot as plt

from surjectors import Chain, MaskedCoupling, Slice, TransformedDistribution
from surjectors.nn import make_mlp
from surjectors.util import (
    as_batch_iterator,
    as_model,
    make_alternating_binary_mask,
)
```

### How to construct a Haiku module

We build a normalizing flow using Haiku modules; a fuller Haiku introduction is [here](https://dm-haiku.readthedocs.io/en/latest/).

```python
def make_flow(n_dimensions):
    def build_td():
        def decoder_fn(n_dim):
            def _fn(z):
                params = make_mlp([4, n_dim * 2])(z)
                means, log_scales = jnp.split(params, 2, -1)
                return distrax.Independent(
                    distrax.Normal(means, jnp.exp(log_scales)),
                    reinterpreted_batch_ndims=1,
                )

            return _fn

        transform = Slice(n_dimensions // 2, decoder_fn(n_dimensions // 2))
        base_distribution = distrax.Independent(
            distrax.Normal(
                jnp.zeros(n_dimensions // 2), jnp.ones(n_dimensions // 2)
            ),
            reinterpreted_batch_ndims=1,
        )
        return TransformedDistribution(base_distribution, transform)

    return as_model(build_td, methods=("log_prob", "sample"))
```

This example uses a `Slice` surjector to reduce the dimensionality of some data by a factor of two: ` Slice(n_dimensions // 2, decoder_fn(n_dimensions // 2))`. `as_model` wraps the zero-argument builder in `hk.transform` and returns an `hk.MultiTransformed` pair whose `apply` exposes each named method (here `log_prob` and `sample`) as its own typed callable — see the next section for why.

With Haiku, we first initialize trainable parameters using `fn.init`.

```python
rng_key_seq = hk.PRNGSequence(0)

n, p = 1000, 10
y = jr.normal(next(rng_key_seq), (n, p))

fn = make_flow(p)
params = fn.init(next(rng_key_seq), y=y)
params
```

The flow's only trainable parameters are the MLP weights inside `decoder_fn`; `Slice` itself has none. Let's sample some data first.

```python
samples = fn.apply.sample(params, next(rng_key_seq), sample_shape=(2,))
samples
```

To call a different method, use the matching attribute on `fn.apply`. Computing the log probability of the data can be done with `fn.apply.log_prob`.

```python
fn.apply.log_prob(params, next(rng_key_seq), y=samples)
```

### Working with Haiku modules

Plain `hk.transform` gives a single entry point, so calling more than one method of an object (e.g. both `log_prob` and `sample` on a `TransformedDistribution`) needs `hk.multi_transform` instead — it lets a builder function return several named callables sharing the same parameters. `as_model` wraps this: it takes a zero-argument callable building a `TransformedDistribution` plus the method names to expose, and returns an `hk.MultiTransformed` pair. Initializing uses the first listed method (`log_prob` here); each named method is then reachable as its own typed callable on `model.apply`:

```python
model = as_model(
    lambda: TransformedDistribution(base_distribution, transform),
    methods=("log_prob", "sample"),
)
params = model.init(rng_key, y=y)
model.apply.log_prob(params, rng_key, y=y)
model.apply.sample(params, rng_key, sample_shape=(10,))
```

Which methods each class implements is in the API documentation.

### How to construct `TransformedDistribution` objects

The `TransformedDistribution` class takes a base distribution and a transformation that maps a sample from the base distribution to the distribution that is induced by the transformation. It's important to note that the dimensionality of the base distribution needs to have the same cardinality as the result of applying a transform on a data point. For instance, consider the transform below where we again use a `Slice` surjector.

```python
def make_transform(n_dimensions):
    def decoder_fn(n_dim):
        def fn(z):
            params = make_mlp([4, n_dim * 2])(z)
            means, log_scales = jnp.split(params, 2, -1)
            return distrax.Independent(
                distrax.Normal(means, jnp.exp(log_scales)),
                reinterpreted_batch_ndims=1,
            )

        return fn

    apply_fns = namedtuple("ApplyFns", ("inverse", "forward"))

    def _model():
        slic = Slice(n_dimensions // 2, decoder_fn(n_dimensions // 2))

        def init(y):
            return slic.inverse(y)

        return init, apply_fns(slic.inverse, slic.forward)

    return hk.multi_transform(_model)
```

The transform above defines a simple parameterized transformation and not a pushforward measure (in which we are typically interested in when working with flows). Calling `transform.apply.inverse` reduces the dimensionality of the input; calling `transform.apply.forward` increases it.

```python
rng_key_seq = hk.PRNGSequence(0)

n, p = 1000, 10
y = jr.normal(next(rng_key_seq), (n, p))

transform = make_transform(p)
params = transform.init(next(rng_key_seq), y=y)

z = transform.apply.inverse(params, next(rng_key_seq), y=y)
z.shape
```

The *forward* transformation takes as an argument a variable $Z$ from the latent space:

```python
y_hat = transform.apply.forward(params, next(rng_key_seq), z=z)
y_hat.shape
```

When defining a normalizing flow, it assumes that the transformation is the **inverse** mapping, i.e., the mapping from data to latent space. Hence the transform $f$ is defined $f: \mathcal{Y} \rightarrow \mathcal{Z}$ (technically this means that
`TransformedDistribution` defines the pullback).

With this, we can now define a `TransformedDistribution`. A transformed distribution is defined by

- a base measure `base_distribution`,
- and a transform `transform`.

For instance, again as a Haiku module:

```python
def make_flow(n_dimensions):
    def build_td():
        def decoder_fn(n_dim):
            def fn(z):
                params = make_mlp([64, 64, n_dim * 2])(z)
                means, log_scales = jnp.split(params, 2, -1)
                return distrax.Independent(
                    distrax.Normal(means, jnp.exp(log_scales)),
                    reinterpreted_batch_ndims=1,
                )

            return fn

        transform = Slice(n_dimensions // 2, decoder_fn(n_dimensions // 2))
        base_distribution = distrax.Independent(
            distrax.Normal(
                jnp.zeros(n_dimensions // 2), jnp.ones(n_dimensions // 2)
            ),
            reinterpreted_batch_ndims=1,
        )
        return TransformedDistribution(base_distribution, transform)

    return as_model(build_td, methods=("log_prob", "sample"))
```

Initialize and apply the flow:

```python
rng_key_seq = hk.PRNGSequence(0)

n, p = 1000, 100
y = jr.normal(next(rng_key_seq), (n, p))

fn = make_flow(p)
params = fn.init(next(rng_key_seq), y=y)

lps = fn.apply.log_prob(params, next(rng_key_seq), y=y)
samples = fn.apply.sample(params, next(rng_key_seq), sample_shape=(10,))
```

### Training a normalizing flow

We are now set up to actually train the parameters of the flow which we do using the gradient-transformation library Optax. The code excerpt below can be used for training any flow in `Surjectors` (with some minor modifications).

```python
def train(rng_key, data, model, n_iter=1000):
    # convert the data set to an iterator
    batch_key, rng_key = jr.split(rng_key)
    train_iter = as_batch_iterator(batch_key, data, 100, True)

    # initialize the model
    init_key, rng_key = jr.split(rng_key)
    params = model.init(init_key, **train_iter(0))

    # create an optimizer
    optimizer = optax.adam(1e-4)
    state = optimizer.init(params)

    @jit
    # gradient step
    def step(params, state, **batch):
        def loss_fn(params):
            lp = model.apply.log_prob(params, None, **batch)
            return -jnp.mean(lp)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_state = optimizer.update(grads, state, params)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, new_state

    losses = np.zeros(n_iter)
    # training loop
    for i in range(n_iter):
        train_loss = 0.0
        # iterate over batches
        for j in range(train_iter.num_batches):
            batch = train_iter(j)
            batch_loss, params, state = step(params, state, **batch)
            train_loss += batch_loss
        losses[i] = train_loss

    return params, losses
```

Let's test this to train a flow on a simple two-dimensional data set.

```python
n, p = 1000, 2
z = jr.normal(next(rng_key_seq), (n, p))
y = jnp.vstack([jnp.abs(z[:, 0]), jax.nn.sigmoid(z[:, 1])]).T
```

```python
_, axes = plt.subplots(figsize=(8, 3), ncols=2)
for i, ax in enumerate(axes):
    ax.hist(y[:, i], color="black", density=True, bins=20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
plt.show()
```

![](../figures/introduction_00.png)

We use a masked coupling bijector with affine transformations. Specifically, we use five masked coupling layers that we concatenate using the `Chain` operator.

```python
def make_flow(n_dimensions):
    def build_td():
        def bijector_fn(params):
            means, log_scales = jnp.split(params, 2, -1)
            return distrax.ScalarAffine(means, jnp.exp(log_scales))

        layers = []
        for i in range(5):
            layer = MaskedCoupling(
                mask=make_alternating_binary_mask(n_dimensions, i % 2 == 0),
                bijector_fn=bijector_fn,
                conditioner=make_mlp([32, 32, n_dimensions * 2]),
            )
            layers.append(layer)

        transform = Chain(layers)
        base_distribution = distrax.Independent(
            distrax.Normal(jnp.zeros(n_dimensions), jnp.ones(n_dimensions)),
            reinterpreted_batch_ndims=1,
        )
        return TransformedDistribution(base_distribution, transform)

    return as_model(build_td, methods=("log_prob", "sample"))
```

To train the flow we need to provide a data argument for `as_batch_iterator` which has to be a named tuple. If the tuple has one element, called `y`, an unconditional flow is trained. If a tuple with two elements, called `y` and `x`, is provided, a conditional flow is trained.

```python
fn = make_flow(p)
data = namedtuple("named_dataset", "y")(y)
params, losses = train(next(rng_key_seq), data, fn)
```

That's it. Let's draw some samples.

```python
samples = fn.apply.sample(params, next(rng_key_seq), sample_shape=(1000,))
```

```python
_, axes = plt.subplots(figsize=(8, 3), ncols=2)
for i, ax in enumerate(axes):
    ax.hist(
        y[:, i],
        color="black",
        density=True,
        bins=20,
        alpha=0.2,
        label="Training data",
    )
    ax.hist(
        samples[:, i],
        color="blue",
        density=True,
        bins=20,
        alpha=0.2,
        label="NF data",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
plt.legend(loc="upper right", bbox_to_anchor=(1.5, 0.6))
plt.show()
```

![](../figures/introduction_01.png)
