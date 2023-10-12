:github_url: https://github.com/dirmeier/surjectors

👋 Welcome to Surjectors!
=========================

Surjectors is a light-weight library for density estimation using
inference and generative surjective normalizing flows, i.e., flows can that reduce or increase dimensionality.
Surjectors builds on Distrax and Haiku and is fully compatible with both of them.

Surjectors makes use of

- Haiku`s module system for neural networks,
- Distrax for probability distributions and some base bijectors,
- Optax for gradient-based optimization,
- JAX for autodiff and XLA computation.

Example usage
-------------

You can, for instance, construct a simple neural process like this:

.. code-block:: python

    from jax import random as jr

    from ramsey import NP, MLP
    from ramsey.data import sample_from_sine_function

    def get_neural_process():
        dim = 128
        np = NP(
            decoder=MLP([dim] * 3 + [2]),
            latent_encoder=(
                MLP([dim] * 3), MLP([dim, dim * 2])
            )
        )
        return np

    key = jr.PRNGKey(23)
    data = sample_from_sine_function(key)

    neural_process = get_neural_process()
    params = neural_process.init(key, x_context=data.x, y_context=data.y, x_target=data.x)

The neural process takes a decoder and a set of two latent encoders as arguments. All of these are typically MLPs, but
Ramsey is flexible enough that you can change them, for instance, to CNNs or RNNs. Once the model is defined, you can initialize
its parameters just like in Flax.

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

1) Clone Surjectors and install :code:`hatch` via :code:`pip install hatch`,
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
    :caption: 🎓 Example code
    :maxdepth: 1
    :hidden:

    examples

..  toctree::
    :caption: 🧱 API
    :maxdepth: 1
    :hidden:

    surjectors
    surjectors.nn
    surjectors.util
