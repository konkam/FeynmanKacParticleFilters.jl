using Revise, ExactWrightFisher, Random, Distributions, RCall
using FeynmanKacParticleFilters

R"library(tidyverse)"

Random.seed!(0);
α_vec = [1.2, 1.4, 1.3]
K = length(α_vec)
Pop_size_WF3 = 10
Nparts = 100
time_grid_WF3 = range(0, stop = 1, length = 10)
wfchain = Wright_Fisher_K_dim_exact_trajectory([0.2, 0.4, 0.4], time_grid_WF3, α_vec)
wfobs_WF3 = [rand(Multinomial(Pop_size_WF3, wfchain[:,k])) for k in 1:size(wfchain,2)] |> l -> hcat(l...)
data_WF3 = Dict(zip(time_grid_WF3 , [wfobs_WF3[:,t] for t in 1:size(wfobs_WF3,2)]))


function multinomial_logpotential(obs_vector::AbstractArray{T, 1}) where T <: Real
    function pot(state::AbstractArray{T, 1})  where T <: Real
        return logpdf(Multinomial(sum(obs_vector), state), obs_vector)
    end
    return pot
end

function create_transition_kernels_WF(data, α_vec::AbstractArray{U, 1}) where {T <: Real, U <: Real}
    sα = sum(α_vec)
    function create_Mt(Δt)
        return state -> Wright_Fisher_K_dim_transition_with_t005_approx(state, Δt, α_vec, sα)
    end

    prior = Dirichlet(repeat([1.], K))
    return FeynmanKacParticleFilters.create_transition_kernels(data, create_Mt, prior)
end

data_WF3[time_grid_WF3[2]]

logGt = FeynmanKacParticleFilters.create_potential_functions(data_WF3, multinomial_logpotential)
Mt = create_transition_kernels_WF(data_WF3, α_vec)
RS(W) = rand(Categorical(W), length(W))

Mt[time_grid_WF3[2]](α_vec / sum(α_vec))

logGt[time_grid_WF3[2]](α_vec / sum(α_vec))

pf = FeynmanKacParticleFilters.generic_particle_filtering_logweights(Mt, logGt, Nparts, RS)


pf_adaptive = FeynmanKacParticleFilters.generic_particle_filtering_adaptive_resampling_logweights(Mt, logGt, Nparts, RS)
