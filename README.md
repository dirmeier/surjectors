# surjectors

[![status](http://www.repostatus.org/badges/latest/concept.svg)](http://www.repostatus.org/#concept)
[![ci](https://github.com/dirmeier/surjectors/actions/workflows/ci.yaml/badge.svg)](https://github.com/dirmeier/surjectors/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/dirmeier/surjectors/branch/main/graph/badge.svg)](https://codecov.io/gh/dirmeier/surjectors)
[![codacy]()]()
[![documentation](https://readthedocs.org/projects/surjectors/badge/?version=latest)](https://surjectors.readthedocs.io/en/latest/?badge=latest)
[![version](https://img.shields.io/pypi/v/surjectors.svg?colorB=black&style=flat)](https://pypi.org/project/surjectors/)

> Surjection layers for density estimation with normalizing flows

## About

Surjectors is a light-weight library of inference and generative surjection layers, i.e., layers that reduce dimensionality, for density estimation using normalizing flows.
Surjectors builds on Distrax and Haiku.

## Example usage

TODO

## Installation

Make sure to have a working `JAX` installation. Depending whether you want to use CPU/GPU/TPU,
please follow [these instructions](https://github.com/google/jax#installation).

To install the latest GitHub <RELEASE>, just call the following on the command line:

```bash
pip install git+https://github.com/dirmeier/surjectors@<RELEASE>
```

## Contributing

In order to contribute:

1) Fork and download the forked repository,
2) create a branch with the name of your new feature (something like `issue/fix-bug-related-to-something` or `feature/implement-new-surjector`),
3) install `surjectors` and dev dependencies via `poetry install` (you might want to create a new `conda` or `venv` environment, to not break other dependencies),
4) develop code, commit changes and push it to your branch,
5) create a PR
