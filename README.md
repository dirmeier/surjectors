# surjectors

[![status](http://www.repostatus.org/badges/latest/concept.svg)](http://www.repostatus.org/#concept)
[![ci](https://github.com/dirmeier/surjectors/actions/workflows/ci.yaml/badge.svg)](https://github.com/dirmeier/surjectors/actions/workflows/ci.yaml)
[![version](https://img.shields.io/pypi/v/surjectors.svg?colorB=black&style=flat)](https://pypi.org/project/surjectors/)

> Surjection layers for density estimation with normalizing flows

## About

Surjectors is a light-weight library of inference and generative surjection layers, i.e., layers that reduce or increase dimensionality, for density estimation using normalizing flows.
Surjectors builds on Distrax and Haiku and is fully compatible with both of them.

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

## Author

Simon Dirmeier <a href="mailto:sfyrbnd @ pm me">sfyrbnd @ pm me</a>
