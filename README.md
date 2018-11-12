# FeynmanKacParticleFilters


A package to perform particle filtering (and smoothing) written using the Feynman-Kac formalism.

Useful stochastic differential equation models implemented:
- [Cox-Ingersoll-Ross](https://en.wikipedia.org/wiki/Cox%E2%80%93Ingersoll%E2%80%93Ross_model)
- Wright-Fisher

Outputs:
- Marginal likelihood
- Samples from the filtering distribution

Implemented:
- Bootstrap particle filter with adaptive resampling.

**References:**

- Chopin, N. & Papaspiliopoulos, O. *A concise introduction to Sequential Monte Carlo*, to appear.
- Del Moral, P. (2004). *Feynman-Kac formulae. Genealogical and interacting particle
systems with applications.* Probability and its Applications. Springer Verlag, New
York.
