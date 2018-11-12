function sample_from_filtering_distributions(particle_filter_output, nsamples, index::Integer)
    return particle_filter_output["X"][rand(Categorical(particle_filter_output["W"][:, index]), nsamples), index]
end

function sample_from_filtering_distributions_logweights(particle_filter_output_logweights, nsamples, index::Integer)
    return particle_filter_output_logweights["X"][rand(Categorical(exp.(particle_filter_output_logweights["logW"][:, index])), nsamples), index]
end
