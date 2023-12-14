using AtomsBase
using AtomsCalculators
using AtomsCalculators.AtomsCalculatorsTesting
using EmpiricalPotentials
using ExtXYZ
using Test
using Unitful

fname = joinpath(pkgdir(EmpiricalPotentials), "data", "TiAl-1024.xyz")
data = ExtXYZ.load(fname) |> FastSystem



@testset "SimplePairPotential" begin
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

@testset "ParametricPairPotential" begin
    emin = -1.0u"meV"
    rmin = 3.1u"Å"
    lj = LennardJones(emin, rmin,  13, 13, 6.0u"Å"; parametric=true)
    test_potential_energy(data, lj)
    #test_forces(data, lj) # Needs update to AtomsCalculators
    test_virial(data, lj)

    # Add more tests for correctness
    @test lj.f(ustrip(rmin), lj.parameters) ≈ ustrip(emin)
    @test lj.f(ustrip(rmin)/2^(1//6), lj.parameters) ≈ 0.0

    E_estimate = AtomsCalculators.potential_energy(data, lj, lj.parameters)
    F_estimate = AtomsCalculators.forces(data, lj, lj.parameters)
    V_estimate = AtomsCalculators.virial(data, lj, lj.parameters)

    @test size(E_estimate) == (2,)
    @test size(F_estimate) == (length(data)*3, 2)
    @test size(V_estimate) == (9, 2)

    @test unit(E_estimate[1]) == EmpiricalPotentials.energy_unit(lj)
    @test unit(V_estimate[1,1]) == EmpiricalPotentials.energy_unit(lj)
    @test unit(F_estimate[1,1]) == EmpiricalPotentials.force_unit(lj)
end