using StatsFuns, Distributions

@testset "Test ESS functions" begin
    @test FeynmanKacParticleFilters.ESS(repeat([1], inner = 10)./10) ≈ 10 atol=10.0^(-7)
    @test FeynmanKacParticleFilters.logESS(repeat([1], inner = 10)./10 |> v -> log.(v)) ≈ log(10) atol=10.0^(-7)
end;

@testset "test particle filter algorithm for CIR process" begin

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
    @test Mt[0.1](3) ≈ 8.418659447049441 atol=10.0^(-7)
    @test Mt[0.1](3.1) ≈ 2.1900629888259893 atol=10.0^(-7)
    @test Mt[0.2](3.1) ≈ 2.6844105017153863 atol=10.0^(-7)
    @test Mt[time_grid[3]](3.1) ≈ 1.3897782586244247 atol=10.0^(-7)

    Random.seed!(0)
    pf = FeynmanKacParticleFilters.generic_particle_filtering1D(Mt, Gt, Nparts, RS)

    Random.seed!(0)
    pf_dict = FeynmanKacParticleFilters.generic_particle_filtering(Mt, Gt, Nparts, RS)

    W = pf["W"]
    w = pf["w"]

    @test typeof(pf) == Dict{String,Array{Float64,2}}
    for i in 1:size(W,2)
        @test pf["W"][1,i] ≈ [1.58397e-6, 0.000109003, 0.247537, 0.332939][i] atol = 10^(-6)
    end
    for i in 1:size(w,2)
        @test pf["w"][1,i] ≈ [8.021083116860762e-8, 1.4329312817343978e-6, 0.03624009164218452, 0.005750007892716746][i] atol = 10^(-10)
    end
    marginal_lik_factors = FeynmanKacParticleFilters.marginal_likelihood_factors(pf)
    # println(marginal_lik_factors)
    res = [ 0.005063925135653128, 0.0013145849369714938, 0.014640244207811792, 0.0017270473953601316]
    for k in 1:Nsteps
        @test marginal_lik_factors[k] ≈ res[k] atol=10.0^(-7)
    end
    @test FeynmanKacParticleFilters.marginal_likelihood(pf, FeynmanKacParticleFilters.marginal_likelihood_factors)  ≈ prod(res) atol=10.0^(-7)

    marginal_lik_factors = FeynmanKacParticleFilters.marginal_likelihood_factors(pf_dict)
    # println(marginal_lik_factors)
    res = [ 0.005063925135653128, 0.0013145849369714938, 0.014640244207811792, 0.0017270473953601316]
    for k in 1:Nsteps
        @test marginal_lik_factors[k] ≈ res[k] atol=10.0^(-7)
    end
    @test FeynmanKacParticleFilters.marginal_likelihood(pf_dict, FeynmanKacParticleFilters.marginal_likelihood_factors)  ≈ prod(res) atol=10.0^(-7)

    @test length(FeynmanKacParticleFilters.sample_from_filtering_distributions1D(pf, 10, 2)) == 10
    @test length(FeynmanKacParticleFilters.sample_from_filtering_distributions(pf_dict, 10, 2)) == 10

    Random.seed!(0)
    pf_logweights = FeynmanKacParticleFilters.generic_particle_filtering_logweights1D(Mt, logGt, Nparts, RS)
    Random.seed!(0)
    pf_logweights_dict = FeynmanKacParticleFilters.generic_particle_filtering_logweights(Mt, logGt, Nparts, RS)

    @test typeof(pf_logweights) == Dict{String,Array{Float64,2}}
    @test typeof(pf_logweights_dict) == Dict{String,Any}

    marginal_loglik_factors = FeynmanKacParticleFilters.marginal_loglikelihood_factors(pf_logweights)
    # println(marginal_loglik_factors)
    res = [ -5.285613377888339, -6.634234300460378, -4.223981089726635, -6.361342036441921]
    for k in 1:Nsteps
        @test marginal_loglik_factors[k] ≈ res[k] atol=5*10.0^(-5)
    end
    marginal_loglik_factors = FeynmanKacParticleFilters.marginal_loglikelihood_factors(pf_logweights_dict)
    res = [ -5.285613377888339, -6.634234300460378, -4.223981089726635, -6.361342036441921]
    for k in 1:Nsteps
        @test marginal_loglik_factors[k] ≈ res[k] atol=5*10.0^(-5)
    end

    @test FeynmanKacParticleFilters.marginal_loglikelihood(pf_logweights, FeynmanKacParticleFilters.marginal_loglikelihood_factors) ≈ sum(res) atol=5*10.0^(-5)
    @test length(FeynmanKacParticleFilters.sample_from_filtering_distributions_logweights1D(pf_logweights, 10, 2)) == 10

    @test FeynmanKacParticleFilters.marginal_loglikelihood(pf_logweights_dict, FeynmanKacParticleFilters.marginal_loglikelihood_factors) ≈ sum(res) atol=5*10.0^(-5)
    @test length(FeynmanKacParticleFilters.sample_from_filtering_distributions_logweights(pf_logweights_dict, 10, 2)) == 10
    #

    Random.seed!(0)
    pf_adaptive = FeynmanKacParticleFilters.generic_particle_filtering_adaptive_resampling1D(Mt, Gt, Nparts, RS)

    marginal_lik_factors = FeynmanKacParticleFilters.marginal_likelihood_factors_adaptive_resampling(pf_adaptive)
    # println(marginal_lik_factors)
    res = [0.005063925135653128, 0.0013145849369714936, 0.014640244207811792, 0.0020015945094952942]
    for k in 1:Nsteps
        @test marginal_lik_factors[k] ≈ res[k] atol=10.0^(-7)
    end
    @test FeynmanKacParticleFilters.marginal_likelihood(pf_adaptive, FeynmanKacParticleFilters.marginal_likelihood_factors) ≈ prod(res) atol=10.0^(-7)
    @test length(FeynmanKacParticleFilters.sample_from_filtering_distributions1D(pf_adaptive, 10, 2)) == 10

    Random.seed!(0)
    pf_adaptive_logweights = FeynmanKacParticleFilters.generic_particle_filtering_adaptive_resampling_logweights1D(Mt, logGt, Nparts, RS)
    marginal_loglik_factors = FeynmanKacParticleFilters.marginal_loglikelihood_factors_adaptive_resampling(pf_adaptive_logweights)
    # println(marginal_loglik_factors)
    res = [ -5.285613377888339, -6.634234300460378, -4.223981089726635, -6.213811161313297]
    for k in 1:Nsteps
        @test marginal_loglik_factors[k] ≈ res[k] atol=5*10.0^(-5)
    end
    @test FeynmanKacParticleFilters.marginal_loglikelihood(pf_adaptive_logweights, FeynmanKacParticleFilters.marginal_loglikelihood_factors_adaptive_resampling) ≈ sum(res) atol=5*10.0^(-5)
    @test length(FeynmanKacParticleFilters.sample_from_filtering_distributions_logweights1D(pf_adaptive_logweights, 10, 2)) == 10

end
