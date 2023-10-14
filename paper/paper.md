---
title: 'Surjectors: surjective normalizing flows for density estimation'
tags:
  - Python
  - JAX
  - Density estimation
  - Normalizing flow
  - Machine learning
  - Statistics
authors:
  - name: Simon Dirmeier^[corresponding author]
    affiliation: "1, 2"
affiliations:
  - name: Swiss Data Science Center, Zurich, Switzerland
    index: 1
  - name: ETH Zurich, Zurich, Switzerland
    index: 2
date: 10 October 2023
bibliography: paper.bib
---

# Summary

Normalizing flows [NFs, @papamakarios2021normalizing] are tractable neural density estimators which have in the recent past been applied successfully for, e.g.,
generative modelling [@kingma2018glow,@ping20wave], Bayesian inference [@rezende15flow,@hoffman2019neutra] or simulation-based inference [@papamakarios2019sequential,@dirmeier2023simulation]. `Surjectors` is a Python library in particular
for *surjective*, i.e., dimensionality-reducing normalizing flows (SNFs, @klein2021funnels). `Surjectors` is based on the libraries JAX, Haiku and Distrax [@jax2018github, @deepmind2020jax] and is fully compatible with them.
By virtue of being entirely written in JAX [@jax2018github], `Surjectors` naturally supports usage on either CPU, GPU and TPU.

# Statement of Need

Real-world data are often lying in a high-dimensional ambient space embedded in a lower-dimensional manifold [@fefferman2016testing] which can complicate estimation of probability densities [@dai2020sliced,@klein2021funnels,@nalisnick2018deep].
As a remedy, recently neural density estimators using surjective normalizing flows (SNFs) have been proposed which reduce the dimensionality of the data while still allowing for exact computation of data likelihoods [@klein2021funnels].
While several computational libraries exist that implement *bijective* normalizing flows, i.e., flows that are dimensionality-preserving, currently none exist that efficiently implement dimensionality-reducing flows.

`Surjectors` is a normalizing flow library that implements both bijective and surjective normalizing flows. `Surjectors` is light-weight, conceptually simple to understand if familiar with the JAX ecosystem, and
computationally efficient due to leveraging the XLA compilation and vectorization from JAX.
We additionally make use of several well-established packages within the JAX ecosystem [@jax2018github] and probabilistic deep learning community.
For composing the conditioning networks that NFs facilitate, `Surjectors` uses the deep learning library Haiku [@haiku2020github]. For training and optimisation, we utilize the gradient transformation library
Optax [@deepmind2020jax]. `Surjectors` leverages Distrax [@deepmind2020jax] and TensorFlow probability [@dillon2017tensorflow] for probability distributions and several base bijector implementations.

# Adoption

@dirmeier2023simulation have proposed a novel method for simulation-based inference where they make use autoregressive inference surjections for density estimation and where they
are using `Surjectors` for their implementations.

# References
