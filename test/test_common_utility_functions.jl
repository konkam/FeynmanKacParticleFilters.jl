@testset "Test common utility functions" begin
    @test FeynmanKacParticleFilters.normalise(1:4) == (1:4)/10
end;
