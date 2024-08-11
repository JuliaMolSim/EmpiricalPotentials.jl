
# using Pkg; Pkg.activate(@__DIR__() * "/..")

##

using AtomsBase, StaticArrays, Test, JSON, ForwardDiff, Unitful
using AtomsCalculators, EmpiricalPotentials, AtomsCalculatorsUtilities,
      AtomsBuilder 

using LinearAlgebra: dot, norm, I 
using AtomsCalculators: potential_energy, forces
using EmpiricalPotentials: cutoff_radius, ZBL  
ACT = AtomsCalculatorsUtilities.Testing

##

# generate the LJ potential 
rcut = 7.0u"Å"
V = ZBL(rcut)

##
# TODO - should add a test the checks the correctness of the 
#        potential implementation analytically 

## 
# finite difference calculator tests 

# TODO - not sure what this next test is and what to do about it
# sys = rattle!(bulk(:Si; cubic=true)*2, 0.1u"Å")
# test_energy_forces_virial(sys, sw)

@info("ZBL Finite difference calculator test")

function rand_struct(nrep) 
   sys = rattle!(bulk(:Al, cubic=true) * nrep, 0.1u"Å")
   return randz!(sys, [ :Al => 0.5, :Cu => 0.5 ])
end 

sys = rand_struct(2)
@test potential_energy(sys, V) isa Unitful.Energy
@test forces(sys, V) isa Vector{<: SVector{3, <: Unitful.Force}}

for sys in [ rand_struct(1), rand_struct(2), rand_struct(2) ]
   @test all( ACT.fdtest(sys, V; rattle = 0.01u"Å", verbose=false) )
end
