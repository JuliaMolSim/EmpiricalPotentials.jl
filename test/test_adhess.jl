

using AtomsBase, Unitful, ExtXYZ, AtomsCalculators, AtomsBuilder,
      EmpiricalPotentials, StaticArrays, Test, JSON, ForwardDiff 
using AtomsCalculators.AtomsCalculatorsTesting
using LinearAlgebra: dot, norm, I 
using EmpiricalPotentials: cutoff_radius, StillingerWeber
using AtomsCalculators: potential_energy, forces

EP = EmpiricalPotentials
ACT = AtomsCalculators.AtomsCalculatorsTesting

sw = StillingerWeber()

function rand_Si_struct(nrep = 2, r = 0.1) 
   rattle!(bulk(:Si, cubic=true), r)   
end

function rand_Si_env(args...)
   sys = rand_Si_struct(args...)
   nlist = EmpiricalPotentials.PairList(sys, cutoff_radius(StillingerWeber()))
   Js, Rs, Zs, z0 = EmpiricalPotentials.get_neighbours(sys, sw, nlist, 1)
   return Rs, Zs, z0 
end

##

# preliminary finit-difference test, this is to be incorporated into the 
# FD tests in the test suite in AtomsCalculators

for ntest = 1:10 
   Rs, Zs, z0 = rand_Si_env()
   Us = randn(eltype(Rs), length(Rs))
   Vs = randn(eltype(Rs), length(Rs))
   F(t) = dot( EP.eval_grad_site(sw, Rs + t * Us, Zs, z0)[2], Vs )
   dF0 = dot( EP.ad_site_hessian(sw, Rs, Zs, z0) * Us, Vs )

   @test ACT.fdtest(F, t -> dF0, 0.0; verbose = false )
end

##

using BenchmarkTools

Rs, Zs, z0 = rand_Si_env()

@info("Timing of eval_grad_site")
@btime EP.eval_grad_site($sw, $Rs, $Zs, $z0)
@info("Timing of ad_site_hessian")
@btime EP.ad_site_hessian($sw, $Rs, $Zs, $z0)
