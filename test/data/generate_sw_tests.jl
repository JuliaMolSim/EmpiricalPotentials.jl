# For this script to run, JuLIP must be installed in the default environment
# so it can be stacked. 
# Run this via 
#    julia --project=.. generate_sw_tests.jl

# Note: The JuLIP.StillingerWeber potential was tested against the 
#       implementation in libatoms/QUIP 
#       https://github.com/libAtoms/QUIP

using JuLIP, StaticArrays, JSON
using LinearAlgebra: dot, norm
using JuLIP: AtomicNumber

##

Ntests = 20 
sw_julip = JuLIP.StillingerWeber()
r0 = 2.35
rcut = cutoff(sw_julip)

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

function rand_test() 
   Rs, Zs, z0 = rand_env()
   v = JuLIP.evaluate(sw_julip, Rs, Zs, z0)
   return Dict("Rs" => Rs, "Zs" => Int.(Zs), "z0" => Int(z0), "val" => v)
end

##

tests = [  rand_test() for _ = 1:Ntests ]
D = Dict("r0" => r0, "rcut" => rcut, "tests" => tests,)
JuLIP.save_json(joinpath(@__DIR__(), "test_sw.json"), D)
