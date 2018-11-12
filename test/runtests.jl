println("Testing...")

using Test
using FeynmanKacParticleFilters
@test 1 == 1

include("test_generic_particle_filter_algorithm.jl")
