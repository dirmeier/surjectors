# Dimensionality reduction with surjections

Surjective normalizing flows use dimensionality-reducing transformations instead of dimensionality-preserving bijective ones.
Below we implement several surjective normalizing for a density estimation problem and compare them to a conventional bijective flow.

```python
import distrax
import haiku as hk
import jax
import numpy as np
import optax
import pandas as pd

from collections import namedtuple
from jax import jit
from jax import numpy as jnp
from jax import random as jr
from tqdm import tqdm

from surjectors import (
    Chain,
    LULinear,
    MaskedCoupling,
    MaskedCouplingInferenceFunnel,
    MLPInferenceFunnel,
    TransformedDistribution,
)
from surjectors.nn import make_mlp
from surjectors.util import (
    as_batch_iterator,
    as_model,
    make_alternating_binary_mask,
)
```

We define a training function first that we can use for all density estimation tasks below.

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
    for i in tqdm(range(n_iter)):
        train_loss = 0.0
        # iterate over batches
        for j in range(train_iter.num_batches):
            batch = train_iter(j)
            batch_loss, params, state = step(params, state, **batch)
            train_loss += batch_loss
        losses[i] = train_loss

    return params, losses
```

### Data

We simulate data from a factor model for testing. The data can easily be embedded in a lower dimensional space via a linear transformation.

```python
rng_key_seq = hk.PRNGSequence(0)
```

```python
n_train, n_test = 1000, 200
n = n_train + n_test
p_data, p_latent = 20, 5

z = jr.normal(next(rng_key_seq), (n, p_latent))
W = jr.normal(next(rng_key_seq), (p_data, p_latent)) * 0.1
y = (W @ z.T).T + jr.normal(next(rng_key_seq), (n, p_data)) * 0.1
```

### A bijective baseline

We start with a simple baseline: a masked coupling flow with rational quadratic splines as transforms. The data is not dimensionality-reducing and hence tries to estimate the density on the $20$-dimensional space.

An RQ spline flow requires defining ranges for which we use the lower and upper bounds of the data

```python
range_min, range_max = float(np.min(y)), float(np.max(y))
```

Next we define the conditioner function.

```python
def make_rq_conditioner(event_shape, hidden_sizes, n_bins):
    n_params = 3 * n_bins + 1
    return hk.Sequential(
        [
            make_mlp(hidden_sizes + [event_shape * n_params]),
            hk.Reshape((event_shape,) + (n_params,), preserve_dims=-1),
        ]
    )
```

We create a baseline that uses five masked coupling flows.

```python
def make_baseline(n_dimensions):
    def build_td():
        def bijector_fn(params):
            return distrax.RationalQuadraticSpline(
                params, range_min=range_min, range_max=range_max
            )

        layers = []
        for i in range(5):
            layer = MaskedCoupling(
                mask=make_alternating_binary_mask(n_dimensions, i % 2 == 0),
                conditioner=make_rq_conditioner(n_dimensions, [128, 128], 4),
                bijector_fn=bijector_fn,
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

Training of the baseline is done as follows:

```python
baseline = make_baseline(p_data)
data = namedtuple("named_dataset", "y")(y[:n_train])
params_baseline, _ = train(next(rng_key_seq), data, baseline)
```

### A surjective MLP funnel

As a first surjective flow, we implement a `MLPInferenceFunnel`. The surjection uses a LU decomposition as inner bijector and a conditional probability density parameterized by an MLP as a decoder. We again use a flow of five layers. The first two and the last two are dimensionality-preserving `LULinear` bijections. The layer in the middle is a dimensionality-reducing funnel.

```python
def make_surjective_mlp_funnel(n_dimensions):
    def build_td():
        def decoder_fn(n_dim):
            def fn(z):
                params = make_mlp([32, 32, n_dim * 2])(z)
                mu, log_scale = jnp.split(params, 2, -1)
                return distrax.Independent(
                    distrax.Normal(mu, jnp.exp(log_scale))
                )

            return fn

        n_dim = n_dimensions
        layers = []
        for i in range(5):
            if i == 2:
                layer = MLPInferenceFunnel(
                    n_keep=int(n_dim / 2), decoder=decoder_fn(int(n_dim / 2))
                )
                n_dim = int(n_dim / 2)
            else:
                layer = LULinear(n_dim)
            layers.append(layer)

        transform = Chain(layers)
        base_distribution = distrax.Independent(
            distrax.Normal(jnp.zeros(n_dim), jnp.ones(n_dim)),
            reinterpreted_batch_ndims=1,
        )
        return TransformedDistribution(base_distribution, transform)

    return as_model(build_td, methods=("log_prob", "sample"))
```

```python
surjective_mlp_funnel = make_surjective_mlp_funnel(p_data)
data = namedtuple("named_dataset", "y")(y[:n_train])
params_surjective_mlp_funnel, _ = train(
    next(rng_key_seq), data, surjective_mlp_funnel
)
```

### A surjective affine masked coupling flow

As a second surjector, we implement a `MaskedCouplingInferenceFunnel` with affine transformations. The surjection uses an affine masked coupling layer as inner bijector and a conditional probability density parameterized by an MLP as a decoder. We use the surjection in the middle of five flow layers. The other four are conventional masked coupling flows.

```python
def make_surjective_affine_masked_coupling(n_dimensions):
    def build_td():
        def bijector_fn(params):
            means, log_scales = jnp.split(params, 2, -1)
            return distrax.ScalarAffine(means, jnp.exp(log_scales))

        def decoder_fn(n_dim):
            def fn(z):
                params = make_mlp([32, 32, n_dim * 2])(z)
                mu, log_scale = jnp.split(params, 2, -1)
                return distrax.Independent(
                    distrax.Normal(mu, jnp.exp(log_scale))
                )

            return fn

        n_dim = n_dimensions
        layers = []
        for i in range(5):
            if i == 2:
                layer = MaskedCouplingInferenceFunnel(
                    n_keep=int(n_dim / 2),
                    decoder=decoder_fn(int(n_dim / 2)),
                    conditioner=make_mlp([128, 128, 2 * n_dim]),
                    bijector_fn=bijector_fn,
                )
                n_dim = int(n_dim / 2)
            else:
                layer = MaskedCoupling(
                    mask=make_alternating_binary_mask(n_dim, i % 2 == 0),
                    conditioner=make_mlp([128, 128, 2 * n_dim]),
                    bijector_fn=bijector_fn,
                )
            layers.append(layer)

        transform = Chain(layers)
        base_distribution = distrax.Independent(
            distrax.Normal(jnp.zeros(n_dim), jnp.ones(n_dim)),
            reinterpreted_batch_ndims=1,
        )
        return TransformedDistribution(base_distribution, transform)

    return as_model(build_td, methods=("log_prob", "sample"))
```

```python
surjective_affine_masked_coupling = make_surjective_affine_masked_coupling(
    p_data
)
data = namedtuple("named_dataset", "y")(y[:n_train])
params_surjective_affine_masked_coupling, _ = train(
    next(rng_key_seq), data, surjective_affine_masked_coupling
)
```

### A surjective rational quadratic masked coupling flow

Finally, we implement a `MaskedCouplingInferenceFunnel` with a rational quadratic transformations. The flow is the same as before, but with affine transformations replaced with splines.

```python
def make_surjective_rq_masked_coupling(n_dimensions):
    def build_td():
        def bijector_fn(params):
            return distrax.RationalQuadraticSpline(
                params, range_min=range_min, range_max=range_max
            )

        def decoder_fn(n_dim):
            def fn(z):
                params = make_mlp([32, 32, n_dim * 2])(z)
                mu, log_scale = jnp.split(params, 2, -1)
                return distrax.Independent(
                    distrax.Normal(mu, jnp.exp(log_scale))
                )

            return fn

        n_dim = n_dimensions
        layers = []
        for i in range(5):
            if i == 2:
                layer = MaskedCouplingInferenceFunnel(
                    n_keep=int(n_dim / 2),
                    decoder=decoder_fn(int(n_dim / 2)),
                    conditioner=make_rq_conditioner(n_dim, [128, 128], 4),
                    bijector_fn=bijector_fn,
                )
                n_dim = int(n_dim / 2)
            else:
                layer = MaskedCoupling(
                    mask=make_alternating_binary_mask(n_dim, i % 2 == 0),
                    conditioner=make_rq_conditioner(n_dim, [128, 128], 4),
                    bijector_fn=bijector_fn,
                )
            layers.append(layer)

        transform = Chain(layers)
        base_distribution = distrax.Independent(
            distrax.Normal(jnp.zeros(n_dim), jnp.ones(n_dim)),
            reinterpreted_batch_ndims=1,
        )
        return TransformedDistribution(base_distribution, transform)

    return as_model(build_td, methods=("log_prob", "sample"))
```

```python
surjective_rq_masked_coupling = make_surjective_rq_masked_coupling(p_data)
data = namedtuple("named_dataset", "y")(y[:n_train])
params_surjective_rq_masked_coupling, _ = train(
    next(rng_key_seq), data, surjective_rq_masked_coupling
)
```

### Density comparisons

Having trained the baseline and surjectors, let's compute density estimates of the training and test data sets using the four models.

```python
model_list = [
    baseline,
    surjective_mlp_funnel,
    surjective_affine_masked_coupling,
    surjective_rq_masked_coupling,
]

param_list = [
    params_baseline,
    params_surjective_mlp_funnel,
    params_surjective_affine_masked_coupling,
    params_surjective_rq_masked_coupling,
]

lps = []
for model, params in zip(model_list, param_list):
    lp_training = model.apply.log_prob(params, None, y=y[:n_train])
    lp_test = model.apply.log_prob(params, None, y=y[n_train:])
    lp_training = jnp.mean(lp_training)
    lp_test = jnp.mean(lp_test)
    lps.append(np.array([lp_training, lp_test]))
```

Not so surprisingly the MLP funnel works best on this data set. The baseline that does not reduce dimensionality has the worst performance.

```python
df = pd.DataFrame(lps, columns=["Training density", "Test density"])
df.insert(
    0,
    "Model",
    [
        "Baseline",
        "MLP funnel",
        "Affine masked coupling funnel",
        "RQ masked coupling funnel",
    ],
)
df
```

| Model | Training density | Test density |
| --- | --- | --- |
| Baseline | 2.9739 | 1.5148 |
| MLP funnel | 10.4925 | 10.4877 |
| Affine masked coupling funnel | 10.4815 | 10.2442 |
| RQ masked coupling funnel | 6.8142 | 5.8006 |
