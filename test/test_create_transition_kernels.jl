@testset "Test creation of transition kernels" begin
    Random.seed!(0)

    Δt = 0.1
    δ = 3.
    γ = 2.5
    σ = 4.
    Nobs = 2
    Nsteps = 4
    λ = 1.
    Nparts = 10
    α = δ/2
    β = γ/σ^2

    time_grid = [k*Δt for k in 0:(Nsteps-1)]
    times = [k*Δt for k in 0:(Nsteps-1)]
    X = FeynmanKacParticleFilters.generate_CIR_trajectory(time_grid, 3, δ*1.2, γ/1.2, σ*0.7)
    Y = map(λ -> rand(Poisson(λ), Nobs), X);
    data = zip(times, Y) |> Dict

    function create_Mt(Δt)
        function Mt(X)
            return FeynmanKacParticleFilters.rCIR.(1, Δt, X, δ, γ, σ)
        end
        return Mt
    end
    prior = Gamma(δ/2, σ^2/γ)#parameterisation shape scale

    tr_kernels =  FeynmanKacParticleFilters.create_transition_kernels(data, create_Mt, prior)

    for t in keys(tr_kernels)
        @test_nowarn tr_kernels[t](1.2)
    end

    backward_tr_kernels =  FeynmanKacParticleFilters.create_backward_transition_kernels(data, create_Mt, prior)

    for t in keys(backward_tr_kernels)
        @test_nowarn backward_tr_kernels[t](1.2)
    end
end;
