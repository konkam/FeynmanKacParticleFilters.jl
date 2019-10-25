@testset "Test some CIR functions" begin

    @test_nowarn FeynmanKacParticleFilters.rCIR(10, 0.5, 0.2, 1., 0.5, 1.2)

    @test_nowarn FeynmanKacParticleFilters.CIR_transition_density(0.2, 0.5, 1.1, 1.2, 0.3, 0.6)
    @test exp(FeynmanKacParticleFilters.CIR_transition_logdensity(0.2, 0.5, 1.1, 1.2, 0.3, 0.6)) ≈ FeynmanKacParticleFilters.CIR_transition_density(0.2, 0.5, 1.1, 1.2, 0.3, 0.6)

    @test_nowarn FeynmanKacParticleFilters.CIR_invariant_density(0.5, 1.2, 0.3, 0.6)
    @test exp(FeynmanKacParticleFilters.CIR_invariant_logdensity(0.5, 1.2, 0.3, 0.6)) ≈ FeynmanKacParticleFilters.CIR_invariant_density(0.5, 1.2, 0.3, 0.6)


    @test FeynmanKacParticleFilters.CIR_transition_logdensity_param_iacus_cuvq_unstable(5, 6, 7, 8) ≈ FeynmanKacParticleFilters.CIR_transition_logdensity_param_iacus_cuvq_scaled_bessel(5, 6, 7, 8)

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

    backward_tr_kernels = FeynmanKacParticleFilters.create_backward_transition_kernels_CIR(data, δ, γ, σ)

    for t in keys(backward_tr_kernels)
        @test_nowarn backward_tr_kernels[t](1.2)
    end
end;
