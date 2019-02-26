@testset "Test some CIR functions" begin

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

    Mt = FeynmanKacParticleFilters.create_transition_kernels_CIR(data, δ, γ, σ)
    logGt = FeynmanKacParticleFilters.create_log_potential_functions_CIR(data)
    RS(W) = rand(Categorical(W), length(W))


    transition_logdensity_CIR(Xtp1, Xt, Δtp1) = FeynmanKacParticleFilters.CIR_transition_logdensity(Xtp1, Xt, Δtp1, δ, γ, σ)

    res = FeynmanKacParticleFilters.generic_FFBS_algorithm_logweights(Mt, logGt, Nparts, Nparts, RS, transition_logdensity_CIR)

    sum_logweights_res = mapslices(logsumexp, res["logW"], dims = 1)
    for k in sum_logweights_res
        @test k ≈ 0 atol=10^(-13)
    end

end;
