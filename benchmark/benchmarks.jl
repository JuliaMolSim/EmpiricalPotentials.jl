using AtomsBase
using AtomsCalculators
using BenchmarkTools
using EmpiricalPotentials
using ExtXYZ
using Folds: SequentialEx
using NeighbourLists
using Unitful

SUITE = BenchmarkGroup()


## Test PairPotential

SUITE["SimplePairPotential"] = BenchmarkGroup()

fname = joinpath(pkgdir(EmpiricalPotentials), "data", "TiAl-1024.xyz")
data = ExtXYZ.load(fname) |> FastSystem

lj = LennardJones(-1.0u"meV", 3.1u"Å",  13, 13, 6.0u"Å")

nlist = PairList(data, cutoff_radius(lj))


SUITE["SimplePairPotential"]["energy"] = BenchmarkGroup()
SUITE["SimplePairPotential"]["energy_forces"] = BenchmarkGroup()

SUITE["SimplePairPotential"]["energy"]["no nlist given"] = @benchmarkable AtomsCalculators.potential_energy($data, $lj)
SUITE["SimplePairPotential"]["energy"]["nlist given"] = @benchmarkable AtomsCalculators.potential_energy($data, $lj; nlist=$nlist)

SUITE["SimplePairPotential"]["energy_forces"]["no nlist given"] = @benchmarkable AtomsCalculators.energy_forces($data, $lj)
SUITE["SimplePairPotential"]["energy_forces"]["nlist given"] = @benchmarkable AtomsCalculators.energy_forces($data, $lj; nlist=$nlist)

