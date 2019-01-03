using Distributions

function rCIR(n::Integer, Dt::Real, x0::Real, δ, γ, σ)
    β = γ/σ^2*exp(2*γ*Dt)/(exp(2*γ*Dt)-1)
    if n == 1
        ks = rand(Poisson(γ/σ^2*x0/(exp(2*γ*Dt)-1)))
        return rand(Gamma(ks+δ/2, 1/β))
    else
        ks = rand(Poisson(γ/σ^2*x0/(exp(2*γ*Dt)-1)), n)
        return rand.(Gamma.(ks .+ δ/2, 1/β))
    end
end

function create_transition_kernels_CIR(data, δ, γ, σ)
    function create_Mt(Δt)
        function Mt(X)
    #         Aind = DualOptimalFiltering.indices_from_multinomial_sample_slow(A)
            return rCIR.(1, Δt, X, δ, γ, σ)
        end
        return Mt
    end
    prior = Gamma(δ/2, σ^2/γ)#parameterisation shape scale
    return create_transition_kernels(data, create_Mt, prior)
end

function create_potential_functions_CIR(data)
    function potential(y)
        return x -> prod(pdf.(Poisson(x), y))
    end
    return create_potential_functions(data, potential)
end

function create_log_potential_functions_CIR(data)
    function potential(y)
        return x -> sum(logpdf.(Poisson(x), y))
    end
    return create_potential_functions(data, potential)
end

transition_CIR(n::Integer, Dt::Real, x0::Real, δ, γ, σ) = rCIR(n::Integer, Dt::Real, x0::Real, δ, γ, σ)

function rec_transition_CIR(Dts, x, δ, γ, σ)
    x_new = transition_CIR(1, Dts[1], x[end], δ, γ, σ)
    if length(Dts) == 1
        return Float64[x; x_new]
    else
        return Float64[x; rec_transition_CIR(Dts[2:end], x_new, δ, γ, σ)]
    end
end

function generate_CIR_trajectory(times, x0, δ, γ, σ)
    Dts = diff(times)
    return rec_transition_CIR(Dts, [x0], δ, γ, σ)
end
