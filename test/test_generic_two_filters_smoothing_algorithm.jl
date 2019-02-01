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

    @test_nowarn FeynmanKacParticleFilters.generic_particle_information_filter1D(Mt, Gt, Nparts, RS)

    @test_nowarn FeynmanKacParticleFilters.generic_particle_information_filter_logweights(Mt, logGt, Nparts, RS)

    @test_nowarn FeynmanKacParticleFilters.generic_particle_information_filter_adaptive_resampling_logweights(Mt, logGt, Nparts, RS)


    transition_density_CIR(Xtp1, Xt, Δtp1) = FeynmanKacParticleFilters.CIR_transition_density(Xtp1, Xt, Δtp1, δ, γ, σ)
    CIR_invariant_density(X) = FeynmanKacParticleFilters.CIR_invariant_density(X, δ, γ, σ)

    transition_logdensity_CIR(Xtp1, Xt, Δtp1) = FeynmanKacParticleFilters.CIR_transition_logdensity(Xtp1, Xt, Δtp1, δ, γ, σ)
    CIR_invariant_logdensity(X) = FeynmanKacParticleFilters.CIR_invariant_logdensity(X, δ, γ, σ)

    Random.seed!(0)
    res = FeynmanKacParticleFilters.two_filter_smoothing_algorithm1D(Mt, Gt, 100, RS, transition_density_CIR, CIR_invariant_density)

    Random.seed!(0)
    logres = FeynmanKacParticleFilters.two_filter_smoothing_algorithm_logweights(Mt, logGt, 100, RS, transition_logdensity_CIR, CIR_invariant_logdensity)

    for k in keys(logres["logW_mn"])
        for l in eachindex(logres["logW_mn"][k])
            @test exp(logres["logW_mn"][k][l]) ≈ res["W_mn"][k][l]
        end
    end

    @test_nowarn FeynmanKacParticleFilters.two_filter_smoothing_algorithm_adaptive_resampling_logweights(Mt, logGt, 100, RS, transition_logdensity_CIR, CIR_invariant_logdensity)

    Random.seed!(0)

    res =  FeynmanKacParticleFilters.two_filter_marginal_smoothing_algorithm1D(Mt, Gt, 100, RS, transition_density_CIR, CIR_invariant_density)

    Random.seed!(0)

    logres =  FeynmanKacParticleFilters.two_filter_marginal_smoothing_algorithm_logweights(Mt, logGt, 100, RS, transition_logdensity_CIR, CIR_invariant_logdensity)

    for k in keys(logres["logW"])
        for l in eachindex(logres["logW"][k])
            @test exp(logres["logW"][k][l]) ≈ res["W"][k][l]
        end
    end

    @test_nowarn  FeynmanKacParticleFilters.two_filter_marginal_smoothing_algorithm_adaptive_resampling_logweights(Mt, logGt, 100, RS, transition_logdensity_CIR, CIR_invariant_logdensity)

end


# @testset "test general particle filter algorithm for CIR process 1 obs" begin
#
#     Random.seed!(0)
#
#     Δt = 0.1
#     δ = 3.
#     γ = 2.5
#     σ = 4.
#     Nobs = 2
#     Nsteps = 4
#     λ = 1.
#     Nparts = 10
#     α = δ/2
#     β = γ/σ^2
#
#     time_grid = [k*Δt for k in 0:(Nsteps-1)]
#     X = FeynmanKacParticleFilters.generate_CIR_trajectory(time_grid, 3, δ*1.2, γ/1.2, σ*0.7)
#     Y = map(λ -> rand(Poisson(λ), Nobs), X);
#     data = zip(time_grid, Y) |> Dict
#     # Y = map(λ -> rand(Poisson(λ), Nobs) |> xx -> [[k] for k in xx], X);
#
#     function create_transition_kernels_CIR_special(data, δ, γ, σ)
#         times = data |> keys |> collect |> sort
#         function create_Mt(Δt)
#             function Mt(X)
#         #         Aind = DualOptimalFiltering.indices_from_multinomial_sample_slow(A)
#                 return (FeynmanKacParticleFilters.rCIR(1, Δt, X[1], δ, γ, σ),)
#             end
#             return Mt
#         end
#         prior = Gamma(δ/2, σ^2/γ)#parameterisation shape scale
#         prior_rng(x) = (rand(prior),)
#         return zip(times, [prior_rng; [create_Mt(times[k]-times[k-1]) for k in 2:length(times)]]) |> Dict
#
#     end
#
#     function create_potential_functions_CIR_special(data)
#         function potential(y)
#             return X -> prod(pdf.(Poisson(X[1]), y))
#         end
#         return FeynmanKacParticleFilters.create_potential_functions(data, potential)
#     end
#
#     function create_log_potential_functions_CIR_special(data)
#         function potential(y)
#             return x -> sum(logpdf.(Poisson(x[1]), y))
#         end
#         return FeynmanKacParticleFilters.create_potential_functions(data, potential)
#     end
#
#     Mt = create_transition_kernels_CIR_special(data, δ, γ, σ)
#     Gt = create_potential_functions_CIR_special(data)
#     logGt = create_log_potential_functions_CIR_special(data)
#     RS(W) = rand(Categorical(W), length(W))
#
#     Random.seed!(0)
#     @test Mt[0.1]((3,))[1] ≈ 8.418659447049441 atol=10.0^(-7)
#     @test Mt[0.1]((3.1,))[1] ≈ 2.1900629888259893 atol=10.0^(-7)
#     @test Mt[0.2]((3.1,))[1] ≈ 2.6844105017153863 atol=10.0^(-7)
#     @test Mt[time_grid[3]]((3.1,))[1] ≈ 1.3897782586244247 atol=10.0^(-7)
#
#     @test Gt[0.1]((3,)) ≈ 2.2129511996787992e-8 atol=10.0^(-7)
#     @test Gt[0.1]((3.1,)) ≈ 3.7273708205666865e-8 atol=10.0^(-7)
#     @test Gt[0.2]((3.1,)) ≈ 0.03877426525100398 atol=10.0^(-7)
#     @test Gt[time_grid[3]]((3.1,)) ≈ 0.03877426525100398 atol=10.0^(-7)
#     @test Gt[time_grid[3]]((3.1,)) == prod(pdf.(Poisson(3.1), data[time_grid[3]]))
#
#     @test logGt[time_grid[3]]((3.1,)) == sum(logpdf.(Poisson(3.1), data[time_grid[3]]))
#
#
#     Random.seed!(0)
#     pf = FeynmanKacParticleFilters.generic_particle_filtering(Mt, Gt, Nparts, RS)
#
#     W = pf["W"]
#     w = pf["w"]
#
#     @test typeof(pf) == Dict{String,Any}
#     for i in 1:size(W,2)
#         @test pf["W"][1,i] ≈ [1.58397e-6, 0.000109003, 0.247537, 0.332939][i] atol = 10^(-6)
#     end
#     for i in 1:size(w,2)
#         @test pf["w"][1,i] ≈ [8.021083116860762e-8, 1.4329312817343978e-6, 0.03624009164218452, 0.005750007892716746][i] atol = 10^(-10)
#     end
#
#     pf_logweights = FeynmanKacParticleFilters.generic_particle_filtering_logweights(Mt, logGt, Nparts, RS)
#
#     @test typeof(pf_logweights) == Dict{String,Any}
#
#     pf_adaptive_dict = FeynmanKacParticleFilters.generic_particle_filtering_adaptive_resampling(Mt, Gt, Nparts, RS)
#
#     @test typeof(pf_adaptive_dict) == Dict{String,Any}
#
#     pf_adaptive_logweights_dict = FeynmanKacParticleFilters.generic_particle_filtering_adaptive_resampling_logweights(Mt, logGt, Nparts, RS)
#
#     @test typeof(pf_adaptive_logweights_dict) == Dict{String,Any}
#
# end