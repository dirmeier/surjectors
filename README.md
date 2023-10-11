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

## Documentation

Documentation can be found [here](https://surjectors.readthedocs.io/en/latest/).

## Examples

You can find several self-contained examples on how to use the algorithms in `examples`.

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
`"good first issue" <https://github.com/dirmeier/surjectors/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22>`_.

In order to contribute:

1) Clone Surjectors and install `hatch` via `pip install hatch`,
2) create a new branch locally `git checkout -b feature/my-new-feature` or `git checkout -b issue/fixes-bug`,
3) implement your contribution and ideally a test case,
4) test it by calling `hatch run test` on the (Unix) command line,
5) submit a PR 🙂

## Author

Simon Dirmeier <a href="mailto:sfyrbnd @ pm me">sfyrbnd @ pm me</a>
