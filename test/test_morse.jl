
# using Pkg; Pkg.activate(@__DIR__() * "/..")

##

using AtomsBase, StaticArrays, Test, ForwardDiff, Unitful
using AtomsCalculators, EmpiricalPotentials, AtomsCalculatorsUtilities,
      AtomsBuilder 

using LinearAlgebra: dot, norm, I 
using AtomsCalculators: potential_energy, forces
using EmpiricalPotentials: cutoff_radius, LennardJones 
ACT = AtomsCalculatorsUtilities.Testing

##

# generate the LJ potential 
rcut = 6.0u"Å"
zAl = AtomsBuilder.Chemistry.atomic_number(:Al)
zCu = AtomsBuilder.Chemistry.atomic_number(:Cu)
params = Dict( (zAl, zAl) => (-1.0u"eV", 2.7u"Å", 5.2), 
               (zAl, zCu) => (-1.234u"eV", 3.2u"Å", 4.5), 
               (zCu, zCu) => (-0.345u"eV", 3.0u"Å", 5.0) )
V = Morse(params, rcut)

##
# TODO - should add a test the checks the correctness of the 
#        potential implementation analytically 

## 
# finite difference calculator tests 

# TODO - not sure what this next test is and what to do about it
# sys = rattle!(bulk(:Si; cubic=true)*2, 0.1u"Å")
# test_energy_forces_virial(sys, sw)

@info("Morse Finite difference calculator test")

function rand_struct_morse(nrep) 
   sys = rattle!(bulk(:Al, cubic=true) * nrep, 0.1u"Å")
   return randz!(sys, [ :Al => 0.5, :Cu => 0.5 ])
end 

sys = rand_struct_morse(2)
@test potential_energy(sys, V) isa Unitful.Energy
@test forces(sys, V) isa Vector{<: SVector{3, <: Unitful.Force}}

for sys in [ rand_struct_morse(1), rand_struct_morse(2), rand_struct_morse(2) ]
   @test all( ACT.fdtest(sys, V; rattle = 0.01u"Å", verbose=false) )
end
