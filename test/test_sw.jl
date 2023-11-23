
using EmpiricalPotentials, StaticArrays, Test 
import JuLIP 

using LinearAlgebra: dot, norm
using EmpiricalPotentials: cutoff_radius
using JuLIP: AtomicNumber

##

sw_julip = JuLIP.StillingerWeber()
sw = StillingerWeber()

r0 = 2.35
rcut = cutoff_radius(sw)

function rand_pos() 
   𝐫 = randn(SVector{3, Float64})
   return (0.8*r0 + rand() * (1.2*rcut - 0.8*r0)) * 𝐫 / norm(𝐫)
end

function rand_env() 
   Rs = [ rand_pos() for _ = 1:rand(3:5) ]
   Zs = [ AtomicNumber(:Si) for _ = 1:length(Rs) ]
   z0 = AtomicNumber(:Si)
   return Rs, Zs, z0
end


for ntest = 1:30
   local Rs, Zs, z0 
   Rs, Zs, z0 = rand_env()
   v1 = EmpiricalPotentials.eval_site(sw, Rs, Zs, z0)
   v2 = JuLIP.evaluate(sw_julip, Rs, Zs, z0)
   @test v1 ≈ v2 
end
