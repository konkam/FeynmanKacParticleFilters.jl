function generic_particle_information_filter1D(Mt, Gt, N, RS)

    res = generic_particle_filtering1D(reverse_time_in_dict_except_first(Mt), reverse_time_in_dict(Gt), N, RS)
    return Dict("w" => res["w"][:,end:-1:1], "W" => res["W"][:,end:-1:1], "X" => res["X"][:,end:-1:1])

end

function generic_particle_information_filter_logweights(Mt, logGt, N, RS)

    res =  generic_particle_filtering1D(reverse_time_in_dict_except_first(Mt), reverse_time_in_dict(Gt), N, RS)
    return Dict("logw" => res["logw"][:,end:-1:1], "logW" => res["logW"][:,end:-1:1], "X" => reverse_time_in_dict(res["X"]))

end

function two_filter_smoothing_algorithm1D(Mt, Gt, N, RS, transition_density::Function, γ::Function)
    #For simplicity, γ is the invariant density of the process
    #transition_density(Xt+1, Xt, Δtp1) is the transition density of Xt+1 given Xt
    particle_filter = generic_particle_filtering1D(Mt, Gt, N, RS)
    information_filter = generic_particle_information_filter1D(Mt, Gt, N, RS)

    w_mn = Dict()
    W_mn = Dict()
    Xt = Dict()
    Xtp1 = Dict()
    times = keys(Mt) |> collect |> sort

    for k in 1:(length(times)-1)
        t = times[k]
        Δtp1 = times[k+1] - times[k]
        Xt[t] = particle_filter["X"][:,k]
        Xtp1[t] = information_filter["X"][:,k+1]
        N = length(Xt[t])
        M = length(Xtp1[t])
        w_mn[t] = Array{Float64, 2}(undef, M, N)
        for n in 1:N
            for m in 1:M
                w_mn[t][m, n] = particle_filter["W"][n,k] * information_filter["W"][m,k+1] * transition_density(Xtp1[t][k], Xt[t][k], Δtp1) / γ(Xtp1[t][k])
            end
        end
        W_mn[t] = normalise(w_mn[t])
    end
    return Dict("W_mn" => W_mn, "Xt" => Xt, "Xtp1" => Xtp1)
end
