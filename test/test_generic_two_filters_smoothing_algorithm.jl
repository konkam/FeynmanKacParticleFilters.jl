using StatsFuns, Distributions

@testset "test 1D information filter algorithm for CIR process with vector observations (repeated)" begin

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
    X = FeynmanKacParticleFilters.generate_CIR_trajectory(time_grid, 3, δ*1.2, γ/1.2, σ*0.7)
    Y = map(λ -> rand(Poisson(λ), Nobs), X);
    data = zip(time_grid, Y) |> Dict
    Mt = FeynmanKacParticleFilters.create_transition_kernels_CIR(data, δ, γ, σ)
    Gt = FeynmanKacParticleFilters.create_potential_functions_CIR(data)
    logGt = FeynmanKacParticleFilters.create_log_potential_functions_CIR(data)
    RS(W) = rand(Categorical(W), length(W))

    backward_Mt = FeynmanKacParticleFilters.create_backward_transition_kernels_CIR(data, δ, γ, σ)

    # @test_nowarn FeynmanKacParticleFilters.generic_particle_information_filter1D(backward_Mt, Gt, Nparts, RS)
    @test_nowarn FeynmanKacParticleFilters.generic_particle_information_filter1D(Mt, Gt, Nparts, RS)

    Random.seed!(0)

    @test_nowarn FeynmanKacParticleFilters.generic_particle_information_filter_logweights(backward_Mt, logGt, Nparts, RS)

    Random.seed!(0)

    @test_nowarn FeynmanKacParticleFilters.generic_particle_information_filter_logweightsV2(backward_Mt, logGt, Nparts, RS)

    @test_nowarn FeynmanKacParticleFilters.generic_particle_information_filter_adaptive_resampling_logweights(Mt, logGt, Nparts, RS)


    transition_density_CIR(Xtp1, Xt, Δtp1) = FeynmanKacParticleFilters.CIR_transition_density(Xtp1, Xt, Δtp1, δ, γ, σ)
    CIR_invariant_density(X) = FeynmanKacParticleFilters.CIR_invariant_density(X, δ, γ, σ)

    transition_logdensity_CIR(Xtp1, Xt, Δtp1) = FeynmanKacParticleFilters.CIR_transition_logdensity(Xtp1, Xt, Δtp1, δ, γ, σ)
    CIR_invariant_logdensity(X) = FeynmanKacParticleFilters.CIR_invariant_logdensity(X, δ, γ, σ)

    Random.seed!(0)
    res = FeynmanKacParticleFilters.two_filter_smoothing_algorithm1D(Mt, Gt, 100, RS, transition_density_CIR, CIR_invariant_density)

    for k in keys(res["W_mn"])
        @test sum(res["W_mn"][k]) ≈ 1
    end

    Random.seed!(0)
    logres = FeynmanKacParticleFilters.two_filter_smoothing_algorithm_logweights(Mt, logGt, 100, RS, transition_logdensity_CIR, CIR_invariant_logdensity)

    for k in keys(logres["logW_mn"])
        for l in eachindex(logres["logW_mn"][k])
            @test exp(logres["logW_mn"][k][l]) ≈ res["W_mn"][k][l]
        end
    end

    res =  FeynmanKacParticleFilters.two_filter_smoothing_algorithm_adaptive_resampling_logweights(Mt, logGt, 100, RS, transition_logdensity_CIR, CIR_invariant_logdensity)

    for k in keys(res["logW_mn"])
        @test logsumexp(res["logW_mn"][k]) ≈ 0 atol=10^(-14)
    end

    @test_nowarn FeynmanKacParticleFilters.two_filter_smoothing_algorithm_logweightsV2(Mt, backward_Mt, logGt, logGt, 100, RS, transition_logdensity_CIR, CIR_invariant_logdensity)

    Random.seed!(0)

    res =  FeynmanKacParticleFilters.two_filter_marginal_smoothing_algorithm1D(Mt, Gt, 100, RS, transition_density_CIR, CIR_invariant_density)

    sum_weights_res = sum(res["W"], dims = 1)

    for k in sum_weights_res
        @test k ≈ 1
    end

    Random.seed!(0)

    logres =  FeynmanKacParticleFilters.two_filter_marginal_smoothing_algorithm_logweights(Mt, logGt, 100, RS, transition_logdensity_CIR, CIR_invariant_logdensity)

    for k in keys(logres["logW"])
        for l in eachindex(logres["logW"][k])
            @test exp(logres["logW"][k][l]) ≈ res["W"][k][l]
        end
    end

    @test_nowarn FeynmanKacParticleFilters.two_filter_marginal_smoothing_algorithm_logweightsV2(Mt, backward_Mt, logGt, logGt, 100, RS, transition_logdensity_CIR, CIR_invariant_logdensity)

    res =   FeynmanKacParticleFilters.two_filter_marginal_smoothing_algorithm_adaptive_resampling_logweights(Mt, logGt, 100, RS, transition_logdensity_CIR, CIR_invariant_logdensity)

    sum_logweights_res = mapslices(logsumexp, res["logW"], dims = 1)
    for k in sum_logweights_res
        @test k ≈ 0 atol=10^(-14)
    end

end
