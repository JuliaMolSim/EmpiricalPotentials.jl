using AtomsBase
# using AtomsCalculators.AtomsCalculatorsTesting
using ExtXYZ
using EmpiricalPotentials, StaticArrays, Test, JSON, ForwardDiff 
using LinearAlgebra: dot, norm, I 
using EmpiricalPotentials: cutoff_radius, StillingerWeber
using AtomsCalculators
using AtomsCalculators: potential_energy, forces

##
# old tests

# generate the SW potential 
sw = StillingerWeber()

# load test data 
D = JSON.parsefile(joinpath(@__DIR__(), "data", "test_sw.json"))
tests = D["tests"]

# the argument t should be a tests[i]
function read_test(t::Dict) 
   Rs = SVector{3, Float64}.(t["Rs"])
   Zs = Int.(t["Zs"])
   z0 = Int(t["z0"])
   v = Float64(t["val"])
   return Rs, Zs, z0, v
end


for t in tests
   local Rs, Zs, z0 
   Rs, Zs, z0, v2 = read_test(t)
   v1 = EmpiricalPotentials.eval_site(sw, Rs, Zs, z0)
   @test v1 ≈ v2
end

## 
# Test the gradients 
# this is an ad hov test implementation that should ideally be replaced 
# asap by a generic implementation that can be used for all potentials. 

sw = StillingerWeber()
for t in tests 
   Rs, Zs, z0, _ = read_test(t) 
   Us = randn(eltype(Rs), length(Rs))
   F(t) = EmpiricalPotentials.eval_site(sw, Rs + t * Us, Zs, z0)
   dF(t) = dot(EmpiricalPotentials.eval_grad_site(sw, Rs + t * Us, Zs, z0)[2], Us)
   @test dF(0.0) ≈ ForwardDiff.derivative(F, 0.0)
end


##
# test the hessian - this is via AD so probably doesn't have to be tested. 
# but we can at least test that it evaluates alright 

Rs, Zs, z0, _ = read_test(tests[1])
try 
   Hi = EmpiricalPotentials.hessian_site(sw, Rs, Zs, z0)
   @test true 
catch e
   @info("EmpiricalPotentials.hessian_site(sw,...) threw an error")
   display(e)
   @test false 
end

try 
   blHi = EmpiricalPotentials.block_hessian_site(sw, Rs, Zs, z0)
   @test true
catch e
   @info("EmpiricalPotentials.block_hessian_site(sw,...) threw an error")
   display(e)
   @test false 
end

## 
# finite difference calculator tests 

# @info("SW Finite difference calculator test")

#    # I've commented out this test; we may need to return to it 
#    # at some point. 
# for sys in [ bulk(:Si, cubic=true) * 1, 
#              bulk(:Si, cubic=false) * 2, ] 
#    @test all( ACT.fdtest(sw, sys; rattle = 0.1u"Å", verbose=false) )
# end



## 
# take a brief look at precon 

using LinearAlgebra: eigvals

# convert the block matrix to a conventional matrix 
function convert_blockmat(Pblock)
   Nr = size(Pblock, 1)
   P = zeros(3*Nr, 3*Nr)
   for j1 = 1:Nr, j2 = 1:Nr 
      J1 = 3 * (j1-1) .+ (1:3)
      J2 = 3 * (j2-1) .+ (1:3)
      P[J1, J2] .= Pblock[j1, j2]
   end
   return P 
end

for t in tests 
   Rs, Zs, z0, _ = read_test(t) 
   # set innerstab = 0.1 to make tests independent of that choice 
   Pblock = EmpiricalPotentials.precon(sw, Rs, Zs, z0, 0.1)
   P = convert_blockmat(Pblock)
   @test P' ≈ P 
   # compute spectrum and check it is what we expect 
   λ = eigvals(P)
   # rotation-invariance should give three zero eigenvalues 
   # but with innerstab = 0.1 those become 0.1. The tests show that 
   # this is not true which must mean that the precon is not 
   # rotation-equivariant. This is something to be looked at in the future.

   # but at least all evals must be >= 0.1 enforced by construction 
   @test all(λ .>= 0.1 - sqrt(eps(Float64)))
end

##
# TODO: finite difference tests for the full calculator 

# test_potential_energy(data, sw)
# # test_forces(data, sw) # Needs update for AtomsCalculators
# test_virial(data, sw)

