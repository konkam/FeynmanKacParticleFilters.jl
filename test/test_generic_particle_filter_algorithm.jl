using StatsFuns, Distributions

@testset "Test ESS functions" begin
    @test FeynmanKacParticleFilters.ESS(repeat([1], inner = 10)./10) ≈ 10 atol=10.0^(-7)
    @test FeynmanKacParticleFilters.logESS(repeat([1], inner = 10)./10 |> v -> log.(v)) ≈ log(10) atol=10.0^(-7)
end;
