using AtomsBase
using AtomsCalculators.AtomsCalculatorsTesting
using EmpiricalPotentials
using ExtXYZ
using Test
using Unitful

fname = joinpath(pkgdir(EmpiricalPotentials), "data", "TiAl-1024.xyz")
data = ExtXYZ.load(fname) |> FastSystem

lj = LennardJones(-1.0u"meV", 3.1u"Å",  13, 13, 6.0u"Å")

@testset "Pair potentials" begin
    test_potential_energy(data, lj)
    #test_forces(data, lj) # Needs update to AtomsCalculators
    test_virial(data, lj)

    # Add test for correctness
end