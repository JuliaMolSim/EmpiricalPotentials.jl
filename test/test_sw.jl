
using EmpiricalPotentials, StaticArrays, Test, JSON 
using LinearAlgebra: dot, norm
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

t = tests[1]

for ntest = 1:length(tests)
   local Rs, Zs, z0 
   Rs, Zs, z0, v2 = read_test(tests[ntest])
   v1 = EmpiricalPotentials.eval_site(sw, Rs, Zs, z0)
   @test v1 ≈ v2
end

## 
# Test the gradients 
