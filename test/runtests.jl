using EmpiricalPotentials
using Test

@testset "EmpiricalPotentials.jl" begin
    # Write your tests here.
    include("test_pairpotentials.jl")

    @testset "StillingerWeber" begin
        include("test_sw.jl")
    end
end
