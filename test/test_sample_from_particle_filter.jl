@testset "test sampling from the particle filter algorithm for CIR process" begin

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
    Gt = FeynmanKacParticleFilters.create_potential_functions_CIR(data)
    logGt = FeynmanKacParticleFilters.create_log_potential_functions_CIR(data)
    RS(W) = rand(Categorical(W), length(W))

    Random.seed!(0)
    @test isreal(Mt[0.1](3))# ≈ 8.418659447049441 atol=10.0^(-7)
    @test isreal(Mt[0.1](3.1))# ≈ 2.1900629888259893 atol=10.0^(-7)
    @test isreal(Mt[0.2](3.1))# ≈ 2.6844105017153863 atol=10.0^(-7)
    @test isreal(Mt[time_grid[3]](3.1))# ≈ 1.3897782586244247 atol=10.0^(-7)

    Random.seed!(0)
    pf = FeynmanKacParticleFilters.generic_particle_filtering1D(Mt, Gt, Nparts, RS)

    Random.seed!(0)
    pf_dict = FeynmanKacParticleFilters.generic_particle_filtering(Mt, Gt, Nparts, RS)

    W = pf["W"]
    w = pf["w"]

    # @test typeof(pf) == Dict{String,Array{Float64,2}}
    # for i in 1:size(W,2)
    #     @test pf["W"][1,i] ≈ [1.58397e-6, 0.000109003, 0.247537, 0.332939][i] atol = 10^(-6)
    # end
    # for i in 1:size(w,2)
    #     @test pf["w"][1,i] ≈ [8.021083116860762e-8, 1.4329312817343978e-6, 0.03624009164218452, 0.005750007892716746][i] atol = 10^(-10)
    # end

    @test length(FeynmanKacParticleFilters.sample_from_filtering_distributions1D(pf, 10, 2)) == 10
    @test length(FeynmanKacParticleFilters.sample_from_filtering_distributions(pf_dict, 10, 2)) == 10

    Random.seed!(0)
    pf_logweights = FeynmanKacParticleFilters.generic_particle_filtering_logweights1D(Mt, logGt, Nparts, RS)
    Random.seed!(0)
    pf_logweights_dict = FeynmanKacParticleFilters.generic_particle_filtering_logweights(Mt, logGt, Nparts, RS)

    @test typeof(pf_logweights) == Dict{String,Array{Float64,2}}
    @test typeof(pf_logweights_dict) == Dict{String,Any}

    @test length(FeynmanKacParticleFilters.sample_from_filtering_distributions_logweights1D(pf_logweights, 10, 2)) == 10

    @test length(FeynmanKacParticleFilters.sample_from_filtering_distributions_logweights(pf_logweights_dict, 10, 2)) == 10
    #

    Random.seed!(0)
    pf_adaptive = FeynmanKacParticleFilters.generic_particle_filtering_adaptive_resampling1D(Mt, Gt, Nparts, RS)

    Random.seed!(0)
    pf_adaptive_dict = FeynmanKacParticleFilters.generic_particle_filtering_adaptive_resampling(Mt, Gt, Nparts, RS)

    @test length(FeynmanKacParticleFilters.sample_from_filtering_distributions1D(pf_adaptive, 10, 2)) == 10
    @test length(FeynmanKacParticleFilters.sample_from_filtering_distributions(pf_adaptive_dict, 10, 2)) == 10


    Random.seed!(0)
    pf_adaptive_logweights = FeynmanKacParticleFilters.generic_particle_filtering_adaptive_resampling_logweights1D(Mt, logGt, Nparts, RS)

    @test length(FeynmanKacParticleFilters.sample_from_filtering_distributions_logweights1D(pf_adaptive_logweights, 10, 2)) == 10

    Random.seed!(0)
    pf_adaptive_logweights_dict = FeynmanKacParticleFilters.generic_particle_filtering_adaptive_resampling_logweights(Mt, logGt, Nparts, RS)
    @test length(FeynmanKacParticleFilters.sample_from_filtering_distributions_logweights(pf_adaptive_logweights_dict, 10, 2)) == 10

    transition_logdensity_CIR(Xtp1, Xt, Δtp1) = FeynmanKacParticleFilters.CIR_transition_logdensity(Xtp1, Xt, Δtp1, δ, γ, σ)
    CIR_invariant_logdensity(X) = FeynmanKacParticleFilters.CIR_invariant_logdensity(X, δ, γ, σ)
    pf =    FeynmanKacParticleFilters.two_filter_marginal_smoothing_algorithm_adaptive_resampling_logweights(Mt, logGt, 100, RS, transition_logdensity_CIR, CIR_invariant_logdensity)

    @test_nowarn FeynmanKacParticleFilters.sample_from_filter_logweights(pf, 10, 2)
    @test_nowarn FeynmanKacParticleFilters.sample_from_smoothing_distributions_logweights(pf, 10, 2)


end
