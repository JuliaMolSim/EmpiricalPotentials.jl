
using EmpiricalPotentials, StaticArrays, Test, JSON, ForwardDiff 
using LinearAlgebra: dot, norm, I 
using EmpiricalPotentials: cutoff_radius, StillingerWeber
using JuLIP: AtomicNumber

##

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
   Rs, Zs, z0, _ = read_test(tests[1]) 
   Us = randn(eltype(Rs), length(Rs))
   F(t) = EmpiricalPotentials.eval_site(sw, Rs + t * Us, Zs, z0)
   dF(t) = dot(EmpiricalPotentials.eval_grad_site(sw, Rs + t * Us, Zs, z0), Us)
   @test dF(0.0) ≈ ForwardDiff.derivative(F, 0.0)
end

