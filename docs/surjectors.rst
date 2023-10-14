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


General
-------

.. autosummary::
    TransformedDistribution
    Chain

TransformedDistribution
~~~~~~~~~~~~~~~~~~~~~~~

..  autoclass:: TransformedDistribution
    :members:

Chain
~~~~~

..  autoclass:: Chain
    :members: __init__

Bijective layers
----------------

.. autosummary::
    MaskedAutoregressive
    MaskedCoupling
    Permutation

Autoregressive bijections
~~~~~~~~~~~~~~~~~~~~~~~~~

..  autoclass:: MaskedAutoregressive
    :members: __init__

Coupling bijections
~~~~~~~~~~~~~~~~~~~

..  autoclass:: MaskedCoupling
    :members: __init__

Other bijections
~~~~~~~~~~~~~~~~

..  autoclass:: Permutation
    :members: __init__

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
    :members: __init__


..  autoclass:: AffineMaskedCouplingInferenceFunnel
    :members: __init__

..  autoclass:: RationalQuadraticSplineMaskedCouplingInferenceFunnel
    :members: __init__

Autoregressive inference surjections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

..  autoclass:: MaskedAutoregressiveInferenceFunnel
    :members: __init__

..  autoclass:: AffineMaskedAutoregressiveInferenceFunnel
    :members: __init__

..  autoclass:: RationalQuadraticSplineMaskedAutoregressiveInferenceFunnel
    :members: __init__

Other inference surjections
~~~~~~~~~~~~~~~~~~~~~~~~~~~

..  autoclass:: LULinear
    :members: __init__

..  autoclass:: MLPInferenceFunnel
    :members: __init__

..  autoclass:: Slice
    :members: __init__
