
function marginal_likelihood_factors(particle_filter_output)
    return mean(particle_filter_output["w"], dims = 1) |> vec
end

function marginal_likelihood_factors_adaptive_resampling(particle_filter_output_adaptive_resampling)
    ntimes = size(particle_filter_output_adaptive_resampling["w"],2)
    resampled = particle_filter_output_adaptive_resampling["resampled"]
    w = particle_filter_output_adaptive_resampling["w"]
    res = Array{Float64,1}(undef, ntimes)
    for t in 1:ntimes
        if (resampled[t])
            res[t] = mean(w[:, t])
        else
            res[t] = sum(w[:, t])/sum(w[:, t-1])
        end
    end
    return res
end

function marginal_likelihood(particle_filter_output, marginal_likelihood_factors_function)
    return marginal_likelihood_factors_function(particle_filter_output) |> prod
end

function marginal_loglikelihood_factors(particle_filter_output_logweights)
    logw = particle_filter_output_logweights["logw"]
    N = size(logw,1)
    return vec(mapslices(StatsFuns.logsumexp, logw, dims = 1) .- log(N))
end


function marginal_loglikelihood_factors_adaptive_resampling(particle_filter_output_adaptive_resampling_logweights)
    ntimes = size(particle_filter_output_adaptive_resampling_logweights["logw"], 2)
    N = size(particle_filter_output_adaptive_resampling_logweights["logw"], 1)
    resampled = particle_filter_output_adaptive_resampling_logweights["resampled"]
    logw = particle_filter_output_adaptive_resampling_logweights["logw"]
    res = Array{Float64,1}(undef, ntimes)
    for t in 1:ntimes
        if (resampled[t])
            res[t] = logsumexp(logw[:, t]) - log(N)
        else
            res[t] = logsumexp(logw[:, t]) - logsumexp(logw[:, t-1])
        end
    end
    return res
end

function marginal_loglikelihood(particle_filter_output_logweights, marginal_loglikelihood_factors_fun)
    return marginal_loglikelihood_factors_fun(particle_filter_output_logweights) |> sum
end
