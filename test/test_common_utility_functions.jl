@testset "Test common utility functions" begin
    @test FeynmanKacParticleFilters.normalise(1:4) == (1:4)/10
    @test FeynmanKacParticleFilters.reverse_time_in_dict(Dict(zip((1,2),("a","b")))) == Dict(zip((2,1),("a","b")))
    @test FeynmanKacParticleFilters.reverse_time_in_dict_except_first(Dict(zip((1, 2, 3), ("a","b", "c")))) == Dict(zip((1, 3, 2),("a","b", "c")))
end;
