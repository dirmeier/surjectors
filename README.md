# surjectors

[![status](http://www.repostatus.org/badges/latest/concept.svg)](http://www.repostatus.org/#concept)
[![ci](https://github.com/dirmeier/surjectors/actions/workflows/ci.yaml/badge.svg)](https://github.com/dirmeier/surjectors/actions/workflows/ci.yaml)
[![version](https://img.shields.io/pypi/v/surjectors.svg?colorB=black&style=flat)](https://pypi.org/project/surjectors/)

> Surjection layers for density estimation with normalizing flows

## About

Surjectors is a light-weight library for density estimation using
inference and generative surjective normalizing flows, i.e., flows can that reduce or increase dimensionality.
Surjectors builds on Distrax and Haiku and is fully compatible with both of them.

Surjectors makes use of

- Haiku`s module system for neural networks,
- Distrax for probability distributions and some base bijectors,
- Optax for gradient-based optimization,
- JAX for autodiff and XLA computation.

## Examples

You can, for instance, construct a simple normalizing flow like this:

```python
import distrax
from jax import numpy as jnp
from surjectors import Slice, LULinear, Chain
from surjectors import TransformedDistribution
from surjectors.nn import make_mlp

def decoder_fn(n_dim):
    def _fn(z):
        params = make_mlp([32, 32, n_dim * 2])(z)
        means, log_scales = jnp.split(params, 2, -1)
        return distrax.Independent(distrax.Normal(means, jnp.exp(log_scales)))
    return _fn

base_distribution = distrax.Normal(jnp.zeros(5), jnp.ones(5))
transform = Chain([Slice(10, decoder_fn(10)), LULinear(5)])
pushforward = TransformedDistribution(base_distribution, transform)
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

## Author

Simon Dirmeier <a href="mailto:sfyrbnd @ pm me">sfyrbnd @ pm me</a>
