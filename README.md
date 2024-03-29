# surjectors

[![active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![ci](https://github.com/dirmeier/surjectors/actions/workflows/ci.yaml/badge.svg)](https://github.com/dirmeier/surjectors/actions/workflows/ci.yaml)
[![version](https://img.shields.io/pypi/v/surjectors.svg?colorB=black&style=flat)](https://pypi.org/project/surjectors/)
[![doi](https://joss.theoj.org/papers/10.21105/joss.06188/status.svg)](https://doi.org/10.21105/joss.06188)

> Surjection layers for density estimation with normalizing flows

## About

Surjectors is a light-weight library for density estimation using
inference and generative surjective normalizing flows, i.e., flows can that reduce or increase dimensionality.
Surjectors makes use of

- [Haiku](https://github.com/deepmind/dm-haiku)`s module system for neural networks,
- [Distrax](https://github.com/deepmind/distrax) for probability distributions and some base bijectors,
- [Optax](https://github.com/deepmind/optax) for gradient-based optimization,
- [JAX](https://github.com/google/jax) for autodiff and XLA computation.

## Examples

You can, for instance, construct a simple normalizing flow like this:

```python
import distrax
import haiku as hk
from jax import numpy as jnp, random as jr
from surjectors import Slice, LULinear, Chain
from surjectors import TransformedDistribution
from surjectors.nn import make_mlp

def decoder_fn(n_dim):
    def _fn(z):
        params = make_mlp([32, 32, n_dim * 2])(z)
        means, log_scales = jnp.split(params, 2, -1)
        return distrax.Independent(distrax.Normal(means, jnp.exp(log_scales)))
    return _fn

@hk.without_apply_rng
@hk.transform
def flow(x):
    base_distribution = distrax.Independent(
        distrax.Normal(jnp.zeros(5), jnp.ones(5)), 1
    )
    transform = Chain([Slice(5, decoder_fn(5)), LULinear(5)])
    pushforward = TransformedDistribution(base_distribution, transform)
    return pushforward.log_prob(x)

x = jr.normal(jr.PRNGKey(1), (1, 10))
params = flow.init(jr.PRNGKey(2), x)
lp = flow.apply(params, x)
```

More self-contained examples can be found in [examples](https://github.com/dirmeier/surjectors/tree/main/examples).

## Documentation

Documentation can be found [here](https://surjectors.readthedocs.io/en/latest/).

## Installation

Make sure to have a working `JAX` installation. Depending whether you want to use CPU/GPU/TPU,
please follow [these instructions](https://github.com/google/jax#installation).

To install the package from PyPI, call:

```bash
pip install surjectors
```

To install the latest GitHub <RELEASE>, just call the following on the command line:

```bash
pip install git+https://github.com/dirmeier/surjectors@<RELEASE>
```

## Contributing

Contributions in the form of pull requests are more than welcome. A good way to start is to check out issues labelled
[good first issue](https://github.com/dirmeier/surjectors/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

In order to contribute:

1) Clone `Surjectors` and install `hatch` via `pip install hatch`,
2) create a new branch locally `git checkout -b feature/my-new-feature` or `git checkout -b issue/fixes-bug`,
3) implement your contribution and ideally a test case,
4) test it by calling `hatch run test` on the (Unix) command line,
5) submit a PR 🙂


## Citing Surjectors

If you find our work relevant to your research, please consider citing:

```
@article{dirmeier2024surjectors,
    author = {Simon Dirmeier},
    title = {Surjectors: surjection layers for density estimation with normalizing flows},
    year = {2024},
    journal = {Journal of Open Source Software},
    publisher = {The Open Journal},
    volume = {9},
    number = {94},
    pages = {6188},
    doi = {10.21105/joss.06188}
}
```

## Author

Simon Dirmeier <a href="mailto:sfyrbnd @ pm me">sfyrbnd @ pm me</a>
