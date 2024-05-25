using EmpiricalPotentials
using Test

@testset "EmpiricalPotentials.jl" begin
    @testset "Pair potentials" begin include("test_pairpotentials.jl") end 
    @testset "StillingerWeber" begin include("test_sw.jl") end
end
