:github_url: https://github.com/dirmeier/surjectors/

Surjectors documentation
========================

Surjectors is a light-weight library of inference and generative surjection layers, i.e., layers that reduce or increase dimensionality, for density estimation using normalizing flows.
Surjectors builds on Distrax and Haiku and is fully compatible with both of them.

Building a surjector is easy:

.. code-block:: python

   def make_surjector(n_dimension, n_latent):
      def _conditional_fn(n_dim):
         decoder_net = mlp_conditioner([32, 32, n_dim * 2])

         def _fn(z):
            params = decoder_net(z)
            mu, log_scale = jnp.split(params, 2, -1)
            return distrax.Independent(distrax.Normal(mu, jnp.exp(log_scale)))

       return _fn
      def _flow(method, **kwargs):
         surjector = AffineMaskedCouplingInferenceFunnel(
            n_latent,
            _conditional_fn(n_dimension - n_latent),
            mlp_conditioner([32, 32, n_dimension * 2]),
         )
         td = TransformedDistribution(
            _base_distribution_fn(n_latent), _transformation_fn(n_dimension)
         )
      return td(method, **kwargs)

    td = hk.transform(_flow)
    return td

       return
       )


Installation
------------

To install from PyPI, call:

.. code-block:: bash

    pip install ramsey

To install the latest GitHub <RELEASE>, just call the following on the
command line:

.. code-block:: bash

    pip install git+https://github.com/dirmeier/ramsey@<RELEASE>

See also the installation instructions for `Haiku <https://github.com/deepmind/dm-haiku>`_ and `JAX <https://github.com/google/jax>`_, if
you plan to use Ramsey on GPU/TPU.

Contributing
------------

Contributions in the form of pull requests are more than welcome. A good way to start is to check out issues labelled
`"good first issue" <https://github.com/ramsey-devs/ramsey/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22>`_.

In order to contribute:

1) Install Ramsey and dev dependencies via :code:`pip install -e '.[dev]'`,
2) test your contribution/implementation by calling :code:`tox` on the (Unix) command line before submitting a PR.

License
-------

Ramsey is licensed under a Apache 2.0 License


..  toctree::
    :caption: Tutorials
    :maxdepth: 1
    :hidden:

    notebooks/neural_process

..  toctree::
    :maxdepth: 1
    :caption: Examples
    :hidden:

    examples/attentive_neural_process

..  toctree::
    :caption: API reference
    :maxdepth: 1
    :hidden:

    api
