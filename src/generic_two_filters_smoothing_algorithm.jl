function generic_particle_information_filter1D(Mt, Gt, N, RS)

    res = generic_particle_filtering1D(reverse_time_in_dict_except_first(Mt), reverse_time_in_dict(Gt), N, RS)
    return Dict("w" => res["w"][:,end:-1:1], "W" => res["W"][:,end:-1:1], "X" => res["X"][:,end:-1:1])

end

function generic_particle_information_filter_logweights(Mt, logGt, N, RS)

    res =  generic_particle_filtering_logweights(reverse_time_in_dict_except_first(Mt), reverse_time_in_dict(logGt), N, RS)
    return Dict("logw" => res["logw"][:,end:-1:1], "logW" => res["logW"][:,end:-1:1], "X" => reverse_time_in_dict(res["X"]))

end

function generic_particle_information_filter_adaptive_resampling_logweights(Mt, logGt, N, RS)

    res =  generic_particle_filtering_adaptive_resampling_logweights(reverse_time_in_dict_except_first(Mt), reverse_time_in_dict(logGt), N, RS)
    return Dict("logw" => res["logw"][:,end:-1:1], "logW" => res["logW"][:,end:-1:1], "X" => reverse_time_in_dict(res["X"]))

end


function two_filter_smoothing_algorithm1D(Mt, Gt, N, RS, transition_density::Function, γ::Function)
    #For simplicity, γ is the invariant density of the process
    #transition_density(Xt+1, Xt, Δtp1) is the transition density of Xt+1 given Xt
    particle_filter = generic_particle_filtering1D(Mt, Gt, N, RS)
    information_filter = generic_particle_information_filter1D(Mt, Gt, N, RS)

    W_mn = Dict{Int64, Array{Float64,2}}()
    Xt = Dict{Int64, Array{Float64,1}}()
    Xtp1 = Dict{Int64, Array{Float64,1}}()
    times::Array{Float64,1} = keys(Mt) |> collect |> sort

    for k in 1:(length(times)-1)
        # t = times[k]
        Δtp1 = times[k+1] - times[k]
        Xt[k] = particle_filter["X"][:,k]
        Xtp1[k] = information_filter["X"][:,k+1]
        N = length(Xt[k])
        M = length(Xtp1[k])
        # Could be made faster if the number of particles is constant and equal
        w_mn = Array{Float64, 2}(undef, M, N)
        for n in 1:N
            for m in 1:M
                # w_mn[t][m, n] = particle_filter["W"][n,k] * information_filter["W"][m,k+1] * transition_density(Xtp1[t][k], Xt[t][k], Δtp1) / γ(Xtp1[t][k])
                logw = log(particle_filter["W"][n,k]) + log(information_filter["W"][m,k+1]) + log(transition_density(Xtp1[k][m], Xt[k][n], Δtp1)) - log(γ(Xtp1[k][m]))
                w_mn[m, n] = exp(logw)
            end
        end
        W_mn[k] = normalise(w_mn)
    end
    return Dict("W_mn" => W_mn, "Xt" => Xt, "Xtp1" => Xtp1)
end


function common_part_two_filter_smoother_with_or_without_resampling(transition_logdensity::Function, logγ::Function, times, particle_filter, information_filter)
    #For simplicity, logγ is the invariant logdensity of the process
    #transition_logdensity(Xt+1, Xt, Δtp1) is the transition logdensity of Xt+1 given Xt

    logW_mn = Dict{Int64, Array{Float64,2}}()
    Xt = Dict{Int64, Array{Float64,1}}()
    Xtp1 = Dict{Int64, Array{Float64,1}}()

    for k in 1:(length(times)-1)
        # t = times[k]
        Δtp1 = times[k+1] - times[k]
        Xt[k] = particle_filter["X"][k]
        Xtp1[k] = information_filter["X"][k+1]
        N = length(Xt[k])
        M = length(Xtp1[k])
        # Could be made faster if the number of particles is constant and equal
        logw_mn = Array{Float64, 2}(undef, M, N)
        for n in 1:N
            for m in 1:M
                logw_mn[m, n] = particle_filter["logW"][n,k] + information_filter["logW"][m,k+1] + transition_logdensity(Xtp1[k][m], Xt[k][n], Δtp1) - logγ(Xtp1[k][m])
            end
        end
        logW_mn[k] = logw_mn .- logsumexp(logw_mn)
    end
    return Dict("logW_mn" => logW_mn, "Xt" => Xt, "Xtp1" => Xtp1)
end

function two_filter_smoothing_algorithm_logweights(Mt, logGt, N, RS, transition_logdensity::Function, logγ::Function)
    #For simplicity, logγ is the invariant logdensity of the process
    #transition_logdensity(Xt+1, Xt, Δtp1) is the transition logdensity of Xt+1 given Xt
    particle_filter = generic_particle_filtering_logweights(Mt, logGt, N, RS)
    information_filter = generic_particle_information_filter_logweights(Mt, logGt, N, RS)
    times::Array{Float64,1} = keys(Mt) |> collect |> sort

    return common_part_two_filter_smoother_with_or_without_resampling(transition_logdensity, logγ, times, particle_filter, information_filter)

end

function two_filter_smoothing_algorithm_adaptive_resampling_logweights(Mt, logGt, N, RS, transition_logdensity::Function, logγ::Function)
    #For simplicity, logγ is the invariant logdensity of the process
    #transition_logdensity(Xt+1, Xt, Δtp1) is the transition logdensity of Xt+1 given Xt
    particle_filter = generic_particle_filtering_adaptive_resampling_logweights(Mt, logGt, N, RS)
    information_filter = generic_particle_information_filter_adaptive_resampling_logweights(Mt, logGt, N, RS)
    times::Array{Float64,1} = keys(Mt) |> collect |> sort

    return common_part_two_filter_smoother_with_or_without_resampling(transition_logdensity, logγ, times, particle_filter, information_filter)

end

function two_filter_marginal_smoothing_algorithm1D(Mt, Gt, N, RS, transition_density::Function, γ::Function)
    two_filter_smoother = two_filter_smoothing_algorithm1D(Mt, Gt, N, RS, transition_density, γ)

    # times_except_last = keys(Mt) |> collect |> sort |> x -> x[1:(end-1)]

    ntimes = length(Mt |> keys)

    #Marginal weights are the sum over the Xt+1, i.e. sum of the weights along m, the first dimension
    # W = Dict(zip(1:(ntimes-1), (sum(two_filter_smoother["W_mn"][k]; dims = 1) |> vec for k in 1:(ntimes-1))))
    W = map(k -> sum(two_filter_smoother["W_mn"][k], dims = 1), 1:(ntimes-1)) |>
          x -> hcat(x...)

    return Dict("X" => two_filter_smoother["Xt"], "W" =>  W)
end

function two_filter_marginal_smoothing_algorithm_logweights(Mt, logGt, N, RS, transition_logdensity::Function, logγ::Function)
    two_filter_smoother = two_filter_smoothing_algorithm_logweights(Mt, logGt, N, RS, transition_logdensity, logγ)

    ntimes = length(Mt |> keys)

    #Marginal weights are the sum over the Xt+1, i.e. sum of the weights along m, the first dimension
    logW = map(k -> mapslices(logsumexp, two_filter_smoother["logW_mn"][k], dims = 1), 1:(ntimes-1)) |>
      x -> hcat(x...)

    return Dict("X" => two_filter_smoother["Xt"], "logW" =>  logW)
end

function two_filter_marginal_smoothing_algorithm_adaptive_resampling_logweights(Mt, logGt, N, RS, transition_logdensity::Function, logγ::Function)
    two_filter_smoother = two_filter_smoothing_algorithm_adaptive_resampling_logweights(Mt, logGt, N, RS, transition_logdensity, logγ)

    ntimes = length(Mt |> keys)

    #Marginal weights are the sum over the Xt+1, i.e. sum of the weights along m, the first dimension
    # logW = Dict(zip(1:(ntimes-1), ([logsumexp(two_filter_smoother["logW_mn"][k][:,n]) for n in 1:N] for k in 1:(ntimes-1))))
    logW = map(k -> mapslices(logsumexp, two_filter_smoother["logW_mn"][k], dims = 1), 1:(ntimes-1)) |>
        x -> hcat(x...)

    return Dict("X" => two_filter_smoother["Xt"], "logW" =>  logW)
end
