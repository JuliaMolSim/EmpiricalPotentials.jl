
using Pkg; Pkg.activate(@__DIR__() * "/..")

##

using AtomsBase, StaticArrays, Test, JSON, ForwardDiff, Unitful
using AtomsCalculators, EmpiricalPotentials, AtomsCalculatorsUtilities,
      AtomsBuilder 

using LinearAlgebra: dot, norm, I 
using AtomsCalculators: potential_energy, forces
using EmpiricalPotentials: cutoff_radius, LennardJones 
ACT = AtomsCalculatorsUtilities.Testing

##

# generate the LJ potential 
rcut = 5.0u"Å"
zAl = AtomsBuilder.Chemistry.atomic_number(:Al)
zCu = AtomsBuilder.Chemistry.atomic_number(:Cu)
emins = Dict( (zAl, zAl) => -1.0u"eV", 
              (zAl, zCu) => -1.234u"eV", 
              (zCu, zCu) => -0.345u"eV" )
rmins = Dict( (zAl, zAl) => 2.7u"Å", 
              (zAl, zCu) => 3.2u"Å", 
              (zCu, zCu) => 3.0u"Å" )              
lj = LennardJones(emins, rmins, rcut)

##
# TODO - should add a test the checks the correctness of the 
#        potential implementation analytically 

## 
# finite difference calculator tests 

# TODO - not sure what this next test is and what to do about it
# sys = rattle!(bulk(:Si; cubic=true)*2, 0.1u"Å")
# test_energy_forces_virial(sys, sw)

@info("LJ Finite difference calculator test")

function rand_struct(nrep) 
   sys = rattle!(bulk(:Al, cubic=true) * nrep, 0.1u"Å")
   return randz!(sys, [ :Al => 0.5, :Cu => 0.5 ])
end 

sys = rand_struct(2)
@test potential_energy(sys, lj) isa Unitful.Energy
@test forces(sys, lj) isa Vector{<: SVector{3, <: Unitful.Force}}

for sys in [ rand_struct(1), rand_struct(2), rand_struct(2) ]
   @test all( ACT.fdtest(sys, lj; rattle = 0.01u"Å", verbose=false) )
end

