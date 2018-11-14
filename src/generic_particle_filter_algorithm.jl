using StatsFuns

function generic_particle_filtering1D(Mt, Gt, N, RS)
    times::Array{Float64, 1} = Mt |> keys |> collect |> sort
    # Initialisation
    X = Array{Float64, 2}(undef, N,length(times))
    w = Array{Float64, 2}(undef, N,length(times))
    W = Array{Float64, 2}(undef, N,length(times))
    A = Array{Int64, 1}(undef, N)
    X[:,1] = Mt[times[1]].(1:N)

    w[:,1] =  Gt[times[1]].(X[:,1])
    W[:,1] = w[:,1] |> normalise

    #Filtering
    for t in 2:length(times)
                A::Array{Int64, 1} = RS(W[:,t-1])
        X[:,t] = Mt[times[t]](X[A,t-1])
        w[:,t] =  Gt[times[t]].(X[:,t])
        W[:,t] = w[:,t] |> normalise
    end

    return Dict("w" => w, "W" => W, "X" => X)

end

function generic_particle_filtering(Mt, Gt, N, RS)
    times::Array{Float64, 1} = Mt |> keys |> collect |> sort
    # Initialisation
    # X = Array{Float64, 2}(undef, N,length(times))
    X = Dict()
    w = Array{Float64, 2}(undef, N,length(times))
    W = Array{Float64, 2}(undef, N,length(times))
    A = Array{Int64, 1}(undef, N)
    X[1] = Mt[times[1]].(1:N)

    w[:,1] =  Gt[times[1]](X[1])
    W[:,1] = w[:,1] |> normalise

    #Filtering
    for t in 2:length(times)
        A::Array{Int64, 1} = RS(W[:,t-1])
        X[t] = Mt[times[t]](X[t-1][A])
        w[:,t] =  Gt[times[t]].(X[t])
        W[:,t] = w[:,t] |> normalise
    end

    return Dict("w" => w, "W" => W, "X" => X)

end

function generic_particle_filtering_logweights1D(Mt, logGt, N, RS)
    times::Array{Float64, 1} = Mt |> keys |> collect |> sort
    # Initialisation
    X = Array{Float64, 2}(undef, N,length(times))
    logw = Array{Float64, 2}(undef, N,length(times))
    logW = Array{Float64, 2}(undef, N,length(times))
    A = Array{Int64, 1}(undef, N)
    X[:,1] = Mt[times[1]].(1:N)

    logw[:,1] =  logGt[times[1]].(X[:,1])
    logW[:,1] = logw[:,1] .- StatsFuns.logsumexp(logw[:,1])

    #Filtering
    for t in 2:length(times)
        A::Array{Int64, 1} = RS(exp.(logW[:,t-1]))
        X[:,t] = Mt[times[t]](X[A,t-1])
        logw[:,t] = logGt[times[t]].(X[:,t])
        logW[:,t] = logw[:,t] .- StatsFuns.logsumexp(logw[:,t])
    end

    return Dict("logw" => logw, "logW" => logW, "X" => X)

end

function generic_particle_filtering_logweights(Mt, logGt, N, RS)
    times::Array{Float64, 1} = Mt |> keys |> collect |> sort
    # Initialisation
    X = Dict()
    logw = Array{Float64, 2}(undef, N,length(times))
    logW = Array{Float64, 2}(undef, N,length(times))
    A = Array{Int64, 1}(undef, N)
    X[1] = Mt[times[1]].(1:N)

    logw[:,1] =  logGt[times[1]](X[1])
    logW[:,1] = logw[:,1] .- StatsFuns.logsumexp(logw[:,1])

    #Filtering
    for t in 2:length(times)
                A::Array{Int64, 1} = RS(exp.(logW[:,t-1]))
        X[t] = Mt[times[t]](X[t-1][A])
        logw[:,t] = logGt[times[t]].(X[t])
        logW[:,t] = logw[:,t] .- StatsFuns.logsumexp(logw[:,t])
    end

    return Dict("logw" => logw, "logW" => logW, "X" => X)

end

function indices_from_multinomial_sample_slow(A)
    return [repeat([k], inner = A[k]) for k in 1:length(A)] |> x -> vcat(x...)
end

function ESS(W)
    return 1/(W.^2 |> sum)
end

function logESS(logW)
    return - logsumexp(2*logW)
end

function generic_particle_filtering_adaptive_resampling1D(Mt, Gt, N, RS)
    times::Array{Float64, 1} = Mt |> keys |> collect |> sort
    ESSmin = N/2 # typical choice
    # Initialisation
    X = Array{Float64, 2}(undef, N,length(times))
    w = Array{Float64, 2}(undef, N,length(times))
    W = Array{Float64, 2}(undef, N,length(times))
    A = Array{Int64, 1}(undef, N)
    ŵ = Array{Int64, 1}(undef, N)
    resampled = Array{Bool, 1}(undef, length(times))

    X[:,1] = Mt[times[1]].(1:N)

    w[:,1] =  Gt[times[1]].(X[:,1])
    W[:,1] = w[:,1] |> normalise
    resampled[1] = true #this is intended to make the likelihood computations work

    #Filtering
    for t in 2:length(times)
                if ESS(W[:,t-1]) < ESSmin
            A::Array{Int64, 1} = RS(W[:,t-1])
            ŵ = 1
            resampled[t] = true
        else
            A = 1:N
            ŵ = w[:,t-1]
            resampled[t] = false
        end
        X[:,t] = Mt[times[t]](X[A,t-1])
        w[:,t] =  ŵ .* Gt[times[t]].(X[:,t])
        W[:,t] = w[:,t] |> normalise
    end

    return Dict("w" => w, "W" => W, "X" => X, "resampled" => resampled)
end

function generic_particle_filtering_adaptive_resampling(Mt, Gt, N, RS)
    times::Array{Float64, 1} = Mt |> keys |> collect |> sort
    ESSmin = N/2 # typical choice
    # Initialisation
    X = Dict()
    w = Array{Float64, 2}(undef, N,length(times))
    W = Array{Float64, 2}(undef, N,length(times))
    A = Array{Int64, 1}(undef, N)
    ŵ = Array{Int64, 1}(undef, N)
    resampled = Array{Bool, 1}(undef, length(times))

    X[1] = Mt[times[1]].(1:N)

    w[:,1] =  Gt[times[1]](X[1])
    W[:,1] = w[:,1] |> normalise
    resampled[1] = true #this is intended to make the likelihood computations work

    #Filtering
    for t in 2:length(times)
                if ESS(W[:,t-1]) < ESSmin
            A::Array{Int64, 1} = RS(W[:,t-1])
            ŵ = 1
            resampled[t] = true
        else
            A = 1:N
            ŵ = w[:,t-1]
            resampled[t] = false
        end
        X[t] = Mt[times[t]](X[t-1][A])
        w[:,t] =  ŵ .* Gt[times[t]].(X[t])
        W[:,t] = w[:,t] |> normalise
    end

    return Dict("w" => w, "W" => W, "X" => X, "resampled" => resampled)
end


function generic_particle_filtering_adaptive_resampling_logweights1D(Mt, logGt, N, RS)
    times::Array{Float64, 1} = Mt |> keys |> collect |> sort
    logESSmin = log(N) - log(2) # typical choice
    # Initialisation
    X = Array{Float64, 2}(undef, N,length(times))
    logw = Array{Float64, 2}(undef, N,length(times))
    logW = Array{Float64, 2}(undef, N,length(times))
    A = Array{Int64, 1}(undef, N)
    logŵ = Array{Float64, 1}(undef, N)
    resampled = Array{Bool, 1}(undef, length(times))

    X[:,1] = Mt[times[1]].(1:N)

    logw[:,1] =  logGt[times[1]].(X[:,1])
    logW[:,1] = logw[:,1] .- StatsFuns.logsumexp(logw[:,1])
    resampled[1] = true #this is intended to make the likelihood computations work

    #Filtering
    for t in 2:length(times)
                if logESS(logW[:,t-1]) < logESSmin
            A::Array{Int64, 1} = RS(exp.(logW[:,t-1]))
            logŵ .= 0.
            resampled[t] = true
        else
            A = 1:N
            logŵ = logw[:,t-1]
            resampled[t] = false
        end
        X[:,t] = Mt[times[t]](X[A,t-1])
        logw[:,t] =  logŵ .+ logGt[times[t]].(X[:,t])
        logW[:,t] = logw[:,t] .- StatsFuns.logsumexp(logw[:,t])
    end

    return Dict("logw" => logw, "logW" => logW, "X" => X, "resampled" => resampled)
end

function generic_particle_filtering_adaptive_resampling_logweights(Mt, logGt, N, RS)
    times::Array{Float64, 1} = Mt |> keys |> collect |> sort
    logESSmin = log(N) - log(2) # typical choice
    # Initialisation
    X = Dict()
    logw = Array{Float64, 2}(undef, N,length(times))
    logW = Array{Float64, 2}(undef, N,length(times))
    A = Array{Int64, 1}(undef, N)
    logŵ = Array{Float64, 1}(undef, N)
    resampled = Array{Bool, 1}(undef, length(times))

    X[1] = Mt[times[1]].(1:N)
    logw[:,1] =  logGt[times[1]](X[1])
    logW[:,1] = logw[:,1] .- StatsFuns.logsumexp(logw[:,1])
    resampled[1] = true #this is intended to make the likelihood computations work

    #Filtering
    for t in 2:length(times)
                if logESS(logW[:,t-1]) < logESSmin
            A::Array{Int64, 1} = RS(exp.(logW[:,t-1]))
            logŵ .= 0.
            resampled[t] = true
        else
            A = 1:N
            logŵ = logw[:,t-1]
            resampled[t] = false
        end
        X[t] = Mt[times[t]](X[t-1][A])
        logw[:,t] =  logŵ .+ logGt[times[t]].(X[t])
        logW[:,t] = logw[:,t] .- StatsFuns.logsumexp(logw[:,t])
    end

    return Dict("logw" => logw, "logW" => logW, "X" => X, "resampled" => resampled)
end
