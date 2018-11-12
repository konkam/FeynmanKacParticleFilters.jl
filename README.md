# FeynmanKacParticleFilters


A package to perform particle filtering (and smoothing) written using the Feynman-Kac formalism.

Implemented as an example:
- [Cox-Ingersoll-Ross](https://en.wikipedia.org/wiki/Cox%E2%80%93Ingersoll%E2%80%93Ross_model)

Outputs:
- Marginal likelihood
- Samples from the filtering distribution

Implemented:
- Bootstrap particle filter with adaptive resampling.

# Preliminary notions

The Feynman-Kac formalism allows to formulate different types of particle filters using the same abstract elements.
The input of a generic particle filter are:

- A Feynman-Kac model M_t, G_t, where:  
  - G_t is a weight function which can be evaluated for all values of t  
  - It is possible to simulate from M_0(dx0) and M_t(x_t-1, dxt)  
- The number of particles N  
- The choice of an unbiased resampling scheme (e.g. multinomial), i.e. an algorithm to draw variables <img src="Latex_equations/rs.gif" width="55">

# How to install the package

Press `]` in the Julia interpreter to enter the Pkg mode and input:

```julia
pkg> add https://github.com/konkam/FeynmanKacParticleFilters.jl
```

# How to use the package

We start by simulating some data:

```julia
using FeynmanKacParticleFilters, Distributions, Random

Random.seed!(0)

Δt = 0.1
δ = 3.
γ = 2.5
σ = 4.
Nobs = 2
Nsteps = 4
λ = 1.
Nparts = 10
α = δ/2
β = γ/σ^2

time_grid = [k*Δt for k in 0:(Nsteps-1)]
times = [k*Δt for k in 0:(Nsteps-1)]
X = FeynmanKacParticleFilters.generate_CIR_trajectory(time_grid, 3, δ*1.2, γ/1.2, σ*0.7)
Y = map(λ -> rand(Poisson(λ), Nobs), X);
data = zip(times, Y) |> Dict
```

Now define the (log)potential kernel and the transition kernel for the Cox-Ingersoll-Ross model:

```julia
Mt = FeynmanKacParticleFilters.create_transition_kernels_CIR(data, δ, γ, σ)
Gt = FeynmanKacParticleFilters.create_potential_functions_CIR(data)
logGt = FeynmanKacParticleFilters.create_log_potential_functions_CIR(data)
RS(W) = rand(Categorical(W), length(W))
```

**References:**

- Chopin, N. & Papaspiliopoulos, O. *A concise introduction to Sequential Monte Carlo*, to appear.
- Del Moral, P. (2004). *Feynman-Kac formulae. Genealogical and interacting particle
systems with applications.* Probability and its Applications. Springer Verlag, New
York.
