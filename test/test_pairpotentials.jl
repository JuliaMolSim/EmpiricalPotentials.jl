using AtomsBase
using AtomsCalculators.AtomsCalculatorsTesting
using EmpiricalPotentials
using ExtXYZ
using Test
using Unitful

fname = joinpath(pkgdir(EmpiricalPotentials), "data", "TiAl-1024.xyz")
data = ExtXYZ.load(fname) |> FastSystem



@testset "Pair potentials" begin
    emin = -1.0u"meV"
    rmin = 3.1u"Å"
    lj = LennardJones(emin, rmin,  13, 13, 6.0u"Å")
    test_potential_energy(data, lj)
    #test_forces(data, lj) # Needs update to AtomsCalculators
    test_virial(data, lj)

    # Add more tests for correctness
    @test lj.f(ustrip(rmin)) ≈ ustrip(emin)
    @test lj.f(ustrip(rmin)/2^(1//6)) ≈ 0.0

end