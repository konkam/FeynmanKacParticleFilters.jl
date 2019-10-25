using Distributions, SpecialFunctions

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
            return rCIR.(1, Δt, X, δ, γ, σ)
        end
        return Mt
    end
    prior = Gamma(δ/2, σ^2/γ)#parameterisation shape scale
    return create_transition_kernels(data, create_Mt, prior)
end

function create_backward_transition_kernels_CIR(data, δ, γ, σ)
    function create_Mt(Δt)
        function Mt(X)
    #         Aind = DualOptimalFiltering.indices_from_multinomial_sample_slow(A)
            return rCIR.(1, Δt, X, δ, γ, σ)
        end
        return Mt
    end
    prior = Gamma(δ/2, σ^2/γ)#parameterisation shape scale
    return create_backward_transition_kernels(data, create_Mt, prior)
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

# Iacus, Stefano M. Simulation and inference for stochastic differential equations: with R examples. Springer Science & Business Media, 2009. p.47
function CIR_transition_density_param_iacus_cuvq(c::Real, u::Real, v::Real, q::Real)
    # return c * exp(-(u+v)) * (u/v)^(q/2) * besseli(q, 2*sqrt(u*v))
    return exp(CIR_transition_logdensity_param_iacus_cuvq(c, u, v, q))
end

CIR_transition_logdensity_param_iacus_cuvq(c::Real, u::Real, v::Real, q::Real) = CIR_transition_logdensity_param_iacus_cuvq_scaled_bessel(c::Real, u::Real, v::Real, q::Real)


function CIR_transition_logdensity_param_iacus_cuvq_unstable(c::Real, u::Real, v::Real, q::Real)

    # This turned out not to be exceptionally numerically stable, overflow problem in the besseli function.

    return log(c) -(u+v) + q/2 * log(v/u) + log(besseli(q, 2*sqrt(u*v)))

end


function CIR_transition_logdensity_param_iacus_cuvq_scaled_bessel(c::Real, u::Real, v::Real, q::Real)

    # Maybe a more stable version, solution suggested in https://github.com/JuliaStats/Distributions.jl/issues/808.
    # Benchmarking times seem to show that besseli and besselix*exp(x) are mostly similar

    return log(c) -(u+v) + q/2 * log(v/u) + log(besselix(q, 2*sqrt(u*v))) + 2*sqrt(u*v)
end

function u_iacus(c::Real, x0::Real, θ2::Real, t::Real)
    return c*x0*exp(-θ2*t)
end
function v_iacus(c::Real, y::Real)
    return c*y
end
function q_iacus(θ1::Real, θ3::Real)
    return 2*θ1/θ3^2-1
end
function c_iacus(θ2::Real, θ3::Real, t::Real)
    return 2*θ2/(θ3^2*(1-exp(-θ2*t)))
end
function CIR_transition_density_param_iacus(x::Real, x0::Real, t::Real, θ1::Real, θ2::Real, θ3::Real)
    c = c_iacus(θ2, θ3, t)
    u = u_iacus(c, x0, θ2, t)
    v = v_iacus(c, x)
    q = q_iacus(θ1, θ3)
    return CIR_transition_density_param_iacus_cuvq(c, u, v, q)
end

function CIR_transition_logdensity_param_iacus(x::Real, x0::Real, t::Real, θ1::Real, θ2::Real, θ3::Real)
    c = c_iacus(θ2, θ3, t)
    u = u_iacus(c, x0, θ2, t)
    v = v_iacus(c, x)
    q = q_iacus(θ1, θ3)
    return CIR_transition_logdensity_param_iacus_cuvq(c, u, v, q)
end

CIR_transition_density(x::Real, x0::Real, t::Real, δ::Real, γ::Real, σ::Real) = CIR_transition_density_param_iacus(x::Real, x0::Real, t::Real, δ*σ^2, 2*γ, 2*σ)

CIR_transition_logdensity(x::Real, x0::Real, t::Real, δ::Real, γ::Real, σ::Real) = CIR_transition_logdensity_param_iacus(x::Real, x0::Real, t::Real, δ*σ^2, 2*γ, 2*σ)

CIR_invariant_density(x::Real, δ::Real, γ::Real, σ::Real) = pdf(Gamma(δ/2, γ/σ^2), x)

CIR_invariant_logdensity(x::Real, δ::Real, γ::Real, σ::Real) = logpdf(Gamma(δ/2, γ/σ^2), x)
