println("Testing...")

using Test, Random, Distributions
using FeynmanKacParticleFilters
@test 1 == 1

@time include("test_generic_particle_filter_algorithm.jl")
@time include("test_common_utility_functions.jl")
