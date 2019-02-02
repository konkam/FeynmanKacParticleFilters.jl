function generic_forward_filtering_backward_smoothing_algorithm_logweights(Mt, logGt, M, N, RS, transition_logdensity::Function)
    times::Array{Float64, 1} = Mt |> keys |> collect |> sort
    Ntimes = length(times)
    pf_logweights = generic_particle_filtering_logweights(Mt, logGt, N, RS)

    logWtT = Array{Float64, 2}(undef, M, Ntimes)
    logWt_tp1_Tmn = Array{Float64, 2}(undef, M, N)

    for k in Ntimes:-1:2
        for m in 1:M
            if k == Ntimes
                logWtT[m, k] = pf_logweights["logW"][m, k]
            else
                logWtT[m, k] = logsumexp(logWt_tp1_Tmn[m,:])
            end
        end
        for n in 1:N
            Δt = times[k] - times[k-1]
            logSnt = logsumexp(pf_logweights["logW"][:, k-1] .+ transition_logdensity.(pf_logweights["X"][k][n], pf_logweights["X"][k-1], Δt))
            for m in 1:M
                logWt_tp1_Tmn[m,n] = logWtT[m, k] + pf_logweights["logW"][m, k-1] + transition_logdensity(pf_logweights["X"][k][n], pf_logweights["X"][k-1][m], Δt) - logSnt
            end
        end
    end
    for m in 1:M
        logWtT[m, 1] = logsumexp(logWt_tp1_Tmn[m,:])
    end
    return Dict("logW" => logWtT, "X" => pf_logweights["X"])
end
