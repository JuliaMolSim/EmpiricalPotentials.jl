using EmpiricalPotentials
using Test

@testset "EmpiricalPotentials.jl" begin
    @testset "Lennard Jones" begin include("test_lennardjones.jl") end 
    @testset "Morse" begin include("test_morse.jl") end
    @testset "StillingerWeber" begin include("test_sw.jl") end
end
