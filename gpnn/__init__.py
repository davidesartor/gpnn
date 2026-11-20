"""Stochastic graph networks: neural nets whose nodes are distributions."""

import jax

# Noise generation dominates the cost of a training step on CPU; the default
# threefry generator is about twice as slow as this one.
jax.config.update("jax_default_prng_impl", "unsafe_rbg")
