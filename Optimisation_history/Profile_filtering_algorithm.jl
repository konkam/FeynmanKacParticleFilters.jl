using Revise, FeynmanKacParticleFilters, Distributions, Random, Profile, ProfileView, JLD, ExactWrightFisher, DualOptimalFiltering, PProf

function simulate_WF3_data(;K = 3, Ntimes_WF3 = 10, seed = 4)

    α = ones(K)*0.75
    sα = sum(α)
    Pop_size_WF3 = 15
    time_step_WF3 = 0.1
    time_grid_WF3 = [k*time_step_WF3 for k in 0:(Ntimes_WF3-1)]
    Random.seed!(seed)
    wfchain_WF3 = Wright_Fisher_K_dim_exact_trajectory(rand(Dirichlet(K,0.3)), time_grid_WF3[1:(end-1)], α)
    # wfchain_WF3 = rand(Dirichlet(K,0.3)) |> z-> DualOptimalFiltering.wright_fisher_PD1(z, 1.5, 50, Ntimes_WF3)[:,2:end]
    wfobs_WF3 = [rand(Multinomial(Pop_size_WF3, wfchain_WF3[:,k])) for k in 1:size(wfchain_WF3,2)] |> l -> hcat(l...)

    data_WF3 = Dict(zip(time_grid_WF3 , [wfobs_WF3[:,t:t]' |> collect for t in 1:size(wfobs_WF3,2)]))
    return data_WF3, wfchain_WF3, α
end


data_WF3, wfchain_WF3, α = simulate_WF3_data()
n_parts = 50

logGt = FeynmanKacParticleFilters.create_potential_functions(data_WF3, DualOptimalFiltering.multinomial_logpotential)
Mt = DualOptimalFiltering.create_transition_kernels_WF(data_WF3, α)
RS(W) = rand(Categorical(W), length(W))


@profile pf = FeynmanKacParticleFilters.generic_particle_filtering_adaptive_resampling_logweights(Mt, logGt, n_parts, RS)