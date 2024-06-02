using AtomsBase
using AtomsCalculators
using AtomsCalculators.AtomsCalculatorsTesting
import AtomsCalculatorsUtilities
using EmpiricalPotentials
using ExtXYZ
using FiniteDiff
using Test
using Unitful

using AtomsCalculatorsUtilities.SitePotentials: energy_unit, length_unit, force_unit, cutoff_radius 
using AtomsCalculators: zero_forces, forces, potential_energy           

fname = joinpath(pkgdir(EmpiricalPotentials), "data", "TiAl-1024.xyz")

data = ExtXYZ.load(fname) |> FastSystem
##

@testset "SimplePairPotential" begin
    emin = -1.0u"meV"
    rmin = 3.1u"Å"
    lj = LennardJones(emin, rmin,  13, 13, 6.0u"Å")
    @show energy_unit(lj)
    @show length_unit(lj) 
    @show force_unit(lj)
    @show AtomsCalculators.promote_force_type(data, lj)
    test_energy_forces_virial(data, lj)

    # Add more tests for correctness
    @test lj.f(ustrip(rmin)) ≈ ustrip(emin)
    @test lj.f(ustrip(rmin)/2^(1//6)) ≈ 0.0
end

##

#=
# this is a quick demo of type-instability of the 
# pair potential implementation. 

emin = -1.0u"meV"
rmin = 3.1u"Å"
lj = LennardJones(emin, rmin,  13, 13, 6.0u"Å")

using NeighbourLists
nlist = PairList(data, cutoff_radius(lj))
Js, Rs, Zs, z0 = AtomsCalculatorsUtilities.SitePotentials.get_neighbours(data, lj, nlist, 2)


# @btime AtomsCalculatorsUtilities.SitePotentials.eval_site($lj, $Rs, $Zs, $z0)

@profview let Rs = Rs, Zs = Zs, z0 = z0, lj = lj 
    for nrun = 1:1_000_000 
        AtomsCalculatorsUtilities.SitePotentials.eval_site(lj, Rs, Zs, z0)
    end
end

=#

##

#=
@testset "ParametricPairPotential" begin
    emin = -1.0u"meV"
    rmin = 3.1u"Å"
    lj = LennardJones(emin, rmin,  13, 13, 6.0u"Å"; parametric=true)
    test_energy_forces_virial(data, lj)

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

    @testset "Parameter estimations" begin
        # Generate reference values with FiniteDiff
        #
        # Helper functions for FiniteDiff
        function g(a ; at=data, pp=lj)
            ppp = ParametricPairPotential(
                pp.f,
                a,
                pp.atom_ids,
                pp.cutoff,
                ustrip(pp.zero_energy),
            )
            AtomsCalculators.potential_energy(at, ppp)
        end
        function gv(a ; at=data, pp=lj)
            ppp = ParametricPairPotential(
                pp.f,
                a,
                pp.atom_ids,
                pp.cutoff,
                ustrip(pp.zero_energy),
            )
            v = AtomsCalculators.virial(at, ppp)
            return reinterpret(Float64, v )
        end
        function gf(a ; at=data, pp=lj)
            ppp = ParametricPairPotential(
                pp.f,
                a,
                pp.atom_ids,
                pp.cutoff,
                ustrip(pp.zero_energy),
            )
            f = AtomsCalculators.forces(at, ppp)
            return reinterpret(Float64, f )
        end

        # Energy test
        fd_energy = FiniteDiff.finite_difference_jacobian(g, lj.parameters)
        p_energy = ustrip.( AtomsCalculators.potential_energy(data, lj, lj.parameters) )
        @test all( x-> isapprox(x[1],x[2]; rtol=1e7), zip(fd_energy, p_energy) )

        # Forces test
        fd_forces = FiniteDiff.finite_difference_jacobian(gf, lj.parameters)
        p_forces = ustrip.( AtomsCalculators.forces(data, lj, lj.parameters) )
        @test all( x-> isapprox(x[1],x[2]; rtol=1e7), zip(fd_forces, p_forces) )

        # Virial test
        fd_virial = FiniteDiff.finite_difference_jacobian(gv, lj.parameters)
        p_virial = ustrip.( AtomsCalculators.virial(data, lj, lj.parameters) )
        @test all( x-> isapprox(x[1],x[2]; rtol=1e7), zip(fd_forces, p_forces) )

    end
end

=#