# surjectors

[![ci](https://github.com/dirmeier/surjectors/actions/workflows/ci.yaml/badge.svg)](https://github.com/dirmeier/surjectors/actions/workflows/ci.yaml)
[![version](https://img.shields.io/pypi/v/surjectors.svg?colorB=black&style=flat)](https://pypi.org/project/surjectors/)
[![doi](https://joss.theoj.org/papers/10.21105/joss.06188/status.svg)](https://doi.org/10.21105/joss.06188)

> Surjection layers for density estimation with normalizing flows

## About

Surjectors is a light-weight library for density estimation using inference and
generative surjective normalizing flows, i.e., flows that can *reduce* or
*increase* dimensionality while retaining exact-likelihood training. It builds
on Distrax and Haiku and is fully compatible with both.

Surjectors makes use of

- [Haiku](https://github.com/deepmind/dm-haiku)'s module system for neural networks,
- [Distrax](https://github.com/deepmind/distrax) for probability distributions and base bijectors,
- [Optax](https://github.com/deepmind/optax) for gradient-based optimization,
- [JAX](https://github.com/google/jax) for autodiff and XLA computation.

### Why surjections?

A conventional normalizing flow is a chain of *bijections*: every layer is
invertible and dimensionality-preserving, and the change of variables uses the
Jacobian log-determinant. A **surjection** relaxes invertibility — it may map to
a lower- (inference) or higher-dimensional (generative) space — and replaces the
log-determinant with an exact *likelihood contribution*. This lets a flow reduce
dimensionality as part of the model while still being trained by maximum
likelihood. See [`EXAMPLES.md`](EXAMPLES.md) for a worked comparison against a
bijective baseline.

## Installation

Make sure to have a working `JAX` installation. Depending on whether you want to
use CPU/GPU/TPU, follow [these instructions](https://github.com/google/jax#installation).

To install from PyPI, call:

```bash
pip install surjectors
```

To install the latest GitHub `<RELEASE>`:

```bash
pip install git+https://github.com/dirmeier/surjectors@<RELEASE>
```

## Quickstart

A flow is built from three objects: a **base distribution**, a **transform**
(a bijector, a surjector, or a `Chain` of them), and a **`TransformedDistribution`**
that ties them together and exposes `log_prob`, `sample`, `sample_and_log_prob`
and `inverse_and_log_prob`.

```python
import distrax
import haiku as hk
from jax import numpy as jnp, random as jr
from surjectors import Slice, LULinear, Chain, TransformedDistribution
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

Because everything runs inside `hk.transform`, a `TransformedDistribution` is
called through a single entry point: pass `method="log_prob"`, `method="sample"`,
etc. to `apply`/`init` to select which method to run. See [`EXAMPLES.md`](EXAMPLES.md).

## API reference

All layers implement `forward_and_likelihood_contribution` /
`inverse_and_likelihood_contribution`; bijections additionally expose the
Distrax-compatible `forward_and_log_det` / `inverse_and_log_det`. Import
everything from the top-level `surjectors` namespace.

### Core

| Object | Description |
|--------|-------------|
| `TransformedDistribution` | A base distribution pushed through a transform; exposes `log_prob`, `sample`, `sample_and_log_prob`, `inverse_and_log_prob`. |
| `Chain` | Compose several bijectors/surjectors into a single transform. |

### Bijective layers (dimensionality-preserving)

| Object | Description |
|--------|-------------|
| `MaskedAutoregressive` | Masked autoregressive layer with a MADE conditioner and a configurable inner bijector. |
| `AffineMaskedAutoregressive` | Masked autoregressive layer with an affine inner bijector. |
| `MaskedCoupling` | RealNVP-style masked coupling layer with a configurable inner bijector. |
| `AffineMaskedCoupling` | Masked coupling with an affine inner bijector. |
| `RationalQuadraticSplineMaskedCoupling` | Masked coupling with a rational-quadratic-spline inner bijector. |
| `Permutation` | Permute the dimensions of the input. |
| `LULinear` | Invertible linear transform parameterised by an LU decomposition. |

### Inference surjection layers (dimensionality-reducing)

| Object | Description |
|--------|-------------|
| `Slice` | Drop dimensions, modelling the removed ones with a decoder distribution. |
| `MLPInferenceFunnel` | LU-linear funnel with an MLP-parameterised decoder. |
| `MaskedCouplingInferenceFunnel` | Coupling funnel with a configurable inner bijector and decoder. |
| `AffineMaskedCouplingInferenceFunnel` | Coupling funnel with an affine inner bijector. |
| `RationalQuadraticSplineMaskedCouplingInferenceFunnel` | Coupling funnel with an RQ-spline inner bijector. |
| `MaskedAutoregressiveInferenceFunnel` | Autoregressive funnel with a configurable inner bijector. |
| `AffineMaskedAutoregressiveInferenceFunnel` | Autoregressive funnel with an affine inner bijector. |
| `RationalQuadraticSplineMaskedAutoregressiveInferenceFunnel` | Autoregressive funnel with an RQ-spline inner bijector. |

### Re-exported Distrax bijectors

`ScalarAffine`, `RationalQuadraticSpline`, `Inverse` — convenient as the *inner*
bijector passed to the layers above via a `bijector_fn`.

### `surjectors.nn`

| Object | Description |
|--------|-------------|
| `MADE` | Masked autoregressive density-estimator network (autoregressive conditioner). |
| `make_mlp` | Build an MLP conditioner/decoder network. |
| `make_transformer` | Build a transformer conditioner. |

### `surjectors.util`

| Object | Description |
|--------|-------------|
| `as_batch_iterator` | Turn a labelled dataset (`named_dataset`) into a minibatch iterator. |
| `make_alternating_binary_mask` | Build an alternating boolean mask for coupling layers. |
| `unstack` | Split an array along an axis into a list of arrays. |

## Examples

- [`EXAMPLES.md`](EXAMPLES.md) — narrative walkthroughs: an introduction, un-/conditional
  density estimation on two-moons, and dimensionality reduction with surjections.
- [`examples/`](examples) — runnable command-line scripts.

## Contributing

Contributions in the form of pull requests are more than welcome. A good way to
start is to check out issues labelled
[good first issue](https://github.com/dirmeier/surjectors/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

In order to contribute:

1) Clone `surjectors` and install `uv` from [here](https://docs.astral.sh/uv/getting-started/installation/).
2) Install all dependencies using `uv sync --all-groups`.
3) Install the Git hooks:

   ```bash
   uv run pre-commit install
   uv run gitlint install-hook
   ```
4) Create a new branch locally, e.g. `git checkout -b feature/my-new-feature`.
5) Implement your contribution and ideally a test case.
6) Check your work (see below).
7) Submit a PR 🙂.

### Development commands

The project uses `uv` for everything (there is no `Makefile`):

```bash
uv sync --all-groups                                   # install
uv run pytest                                          # run tests
uv run ruff check surjectors examples                  # lint
uv run ruff check --fix surjectors examples \
  && uv run ruff format surjectors examples            # auto-fix + format
uv run mypy surjectors                                 # type-check
uv run pre-commit run --all-files                      # all pre-commit hooks
```

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

Simon Dirmeier <a href="mailto:simd23 @ pm me">simd23 @ pm me</a>
