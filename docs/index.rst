:github_url: https://github.com/dirmeier/surjectors

👋 Welcome to Surjectors!
=========================

.. div:: sd-text-left sd-font-italic

   Surjection layers for density estimation with normalizing flows

----

Surjectors is a light-weight library for density estimation using
inference and generative surjective normalizing flows, i.e., flows can that reduce or increase dimensionality.
Surjectors builds on Distrax and Haiku and is fully compatible with both of them.

Surjectors makes use of

- `Haiku's <https://github.com/deepmind/dm-haiku>`_ module system for neural networks,
- `Distrax <https://github.com/deepmind/distrax>`_ for probability distributions and some base bijectors,
- `Optax <https://github.com/deepmind/optax>`_ for gradient-based optimization,
- `JAX <https://github.com/google/jax>`_ for autodiff and XLA computation.

Example
-------

You can, for instance, construct a simple normalizing flow like this:

.. code-block:: python

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

The flow is constructed using three objects: a base distribution, a transformation, and a transformed distribution.

Installation
------------

To install from PyPI, call:

.. code-block:: bash

    pip install surjectors

To install the latest GitHub <RELEASE>, just call the following on the
command line:

.. code-block:: bash

    pip install git+https://github.com/dirmeier/surjectors@<RELEASE>

See also the installation instructions for `JAX <https://github.com/google/jax>`_, if
you plan to use Surjectors on GPU/TPU.

Contributing
------------

Contributions in the form of pull requests are more than welcome. A good way to start is to check out issues labelled
`"good first issue" <https://github.com/dirmeier/surjectors/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22>`_.

In order to contribute:

1) Clone :code:`surjectors` and install :code:`uv` from `here <https://docs.astral.sh/uv/getting-started/installation/>`_,
2) install all dependencies using ``uv sync``,
3) create a new branch locally :code:`git checkout -b feature/my-new-feature` or :code:`git checkout -b issue/fixes-bug`,
4) implement your contribution and ideally a test case,
5) test it by calling ``make format``, ``make lints`` and ``make tests`` on the (Unix) command line,
6) submit a PR 🙂

Citing Surjectors
-----------------

.. code-block:: latex

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

License
-------

Surjectors is licensed under the Apache 2.0 License.

..  toctree::
    :maxdepth: 1
    :hidden:

    🏠 Home <self>

..  toctree::
    :caption: 🎓 Tutorials
    :maxdepth: 1
    :hidden:

    Introduction <notebooks/introduction>
    Unconditional and conditional density estimation <notebooks/normalizing_flows>
    Dimensionality reduction using surjectors <notebooks/dimension_reduction>

..  toctree::
    :caption: 🚀 Examples
    :maxdepth: 1
    :hidden:

    Self-contained scripts <examples>

..  toctree::
    :caption: 🧱 API
    :maxdepth: 1
    :hidden:

    surjectors
    surjectors.nn
    surjectors.util
