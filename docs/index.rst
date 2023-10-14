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

- Haiku`s module system for neural networks,
- Distrax for probability distributions and some base bijectors,
- Optax for gradient-based optimization,
- JAX for autodiff and XLA computation.

Example
-------

You can, for instance, construct a simple normalizing flow like this:

    >>> import distrax
    >>> from jax import numpy as jnp
    >>> from surjectors import Slice, LULinear, Chain
    >>> from surjectors import TransformedDistribution
    >>> from surjectors.nn import make_mlp
    >>>
    >>> def decoder_fn(n_dim):
    >>>     def _fn(z):
    >>>         params = make_mlp([32, 32, n_dim * 2])(z)
    >>>         means, log_scales = jnp.split(params, 2, -1)
    >>>         return distrax.Independent(distrax.Normal(means, jnp.exp(log_scales)))
    >>>     return _fn
    >>>
    >>> base_distribution = distrax.Normal(jnp.zeros(5), jnp.ones(1))
    >>> transform = Chain([Slice(10, decoder_fn(10)), LULinear(5)])
    >>> pushforward = TransformedDistribution(base_distribution, transform)

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

1) Clone :code:`Surjectors` and install :code:`hatch` via :code:`pip install hatch`,
2) create a new branch locally :code:`git checkout -b feature/my-new-feature` or :code:`git checkout -b issue/fixes-bug`,
3) implement your contribution and ideally a test case,
4) test it by calling :code:`hatch run test` on the (Unix) command line,
5) submit a PR 🙂

License
-------

Surjectors is licensed under the Apache 2.0 License.

..  toctree::
    :maxdepth: 1
    :hidden:

    🏠 Home <self>
    📰 News <news>

..  toctree::
    :caption: 🎓 Examples
    :maxdepth: 1
    :hidden:

    Introduction <notebooks/introduction>
    Unconditional and conditional density estimation <notebooks/normalizing_flows>
    Dimensionality reduction using surjectors <notebooks/dimension_reduction>
    Self-contained scripts <examples>

..  toctree::
    :caption: 🧱 API
    :maxdepth: 1
    :hidden:

    surjectors
    surjectors.nn
    surjectors.util
