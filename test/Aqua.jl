using Test
using Aqua
using Factotum

@testset "Aqua.jl" begin
    Aqua.test_all(Factotum)
end
