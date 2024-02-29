``surjectors``
==============

.. currentmodule:: surjectors

----

Normalizing flows have from a computational perspective three components:

- A base distribution for which we use the probability distributions from `Distrax <https://github.com/google-deepmind/distrax>`_.
- A forward transformation $f$ whose Jacobian determinant can be evaluated efficiently. These are the bijectors and surjectors below.
- A transformed distribution that represents the pushforward from a base distribution to the distribution induced by the transformation.

Hence, every normalizing flow can be composed by defining these three components. See an example below.

    >>> import distrax
    >>> from jax import random as jr, numpy as jnp
    >>> from surjectors import Slice, LULinear, Chain
    >>> from surjectors import TransformedDistribution
    >>>
    >>> def decoder_fn(n_dim):
    >>>     def _fn(z):
    >>>         params = make_mlp([4, 4, n_dim * 2])(z)
    >>>         mu, log_scale = jnp.split(params, 2, -1)
    >>>         return distrax.Independent(distrax.Normal(mu, jnp.exp(log_scale)))
    >>>     return _fn
    >>>
    >>> base_distribution = distrax.Normal(jno.zeros(5), jnp.ones(1))
    >>> flow = Chain([Slice(10, decoder_fn(10)), LULinear(5)])
    >>> pushforward = TransformedDistribution(base_distribution, flow)

Regardless of how the chain of transformations (called :code:`flow` above) is defined,
each pushforward has access to four methods :code:`sample`, :code:`sample_and_log_prob`:code:`log_prob`, and :code:`inverse_and_log_prob`.

The exact method declarations can be found in the API below.

General
-------

.. autosummary::
    TransformedDistribution
    Chain

TransformedDistribution
~~~~~~~~~~~~~~~~~~~~~~~

..  autoclass:: TransformedDistribution
    :members: log_prob, sample, inverse_and_log_prob, sample_and_log_prob

Chain
~~~~~

..  autoclass:: Chain
    :members:

Bijective layers
----------------

.. autosummary::
    MaskedAutoregressive
    AffineMaskedAutoregressive
    MaskedCoupling
    AffineMaskedCoupling
    RationalQuadraticSplineMaskedCoupling
    Permutation

Autoregressive bijections
~~~~~~~~~~~~~~~~~~~~~~~~~

..  autoclass:: MaskedAutoregressive
    :members:

..  autoclass:: AffineMaskedAutoregressive
    :members:

Coupling bijections
~~~~~~~~~~~~~~~~~~~

..  autoclass:: MaskedCoupling
    :members:

..  autoclass:: AffineMaskedCoupling
    :members:

..  autoclass:: RationalQuadraticSplineMaskedCoupling
    :members:

Other bijections
~~~~~~~~~~~~~~~~

..  autoclass:: Permutation
    :members:

Inference surjection layers
---------------------------

.. autosummary::
    MaskedCouplingInferenceFunnel
    AffineMaskedCouplingInferenceFunnel
    RationalQuadraticSplineMaskedCouplingInferenceFunnel

    MaskedAutoregressiveInferenceFunnel
    AffineMaskedAutoregressiveInferenceFunnel
    RationalQuadraticSplineMaskedAutoregressiveInferenceFunnel

    LULinear
    MLPInferenceFunnel
    Slice


Coupling inference surjections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

..  autoclass:: MaskedCouplingInferenceFunnel
    :members:

..  autoclass:: AffineMaskedCouplingInferenceFunnel
    :members:

..  autoclass:: RationalQuadraticSplineMaskedCouplingInferenceFunnel
    :members:

Autoregressive inference surjections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

..  autoclass:: MaskedAutoregressiveInferenceFunnel
    :members:

..  autoclass:: AffineMaskedAutoregressiveInferenceFunnel
    :members:

..  autoclass:: RationalQuadraticSplineMaskedAutoregressiveInferenceFunnel
    :members:

Other inference surjections
~~~~~~~~~~~~~~~~~~~~~~~~~~~

..  autoclass:: LULinear
    :members:

..  autoclass:: MLPInferenceFunnel
    :members:

..  autoclass:: Slice
    :members:
