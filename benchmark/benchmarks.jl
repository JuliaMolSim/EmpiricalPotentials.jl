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
SUITE["ParametricPairPotentialPairPotential"] = BenchmarkGroup()

fname = joinpath(pkgdir(EmpiricalPotentials), "data", "TiAl-1024.xyz")
data = ExtXYZ.load(fname) |> FastSystem

lj = LennardJones(-1.0u"meV", 3.1u"Å",  13, 13, 6.0u"Å")
ljp = LennardJones(-1.0u"meV", 3.1u"Å",  13, 13, 6.0u"Å"; parametric=true)

nlist = PairList(data, cutoff_radius(lj))


SUITE["SimplePairPotential"]["energy"] = BenchmarkGroup()
SUITE["SimplePairPotential"]["energy_forces"] = BenchmarkGroup()

SUITE["SimplePairPotential"]["energy"]["no nlist given"] = @benchmarkable AtomsCalculators.potential_energy($data, $lj)
SUITE["SimplePairPotential"]["energy"]["nlist given"] = @benchmarkable AtomsCalculators.potential_energy($data, $lj; nlist=$nlist)

SUITE["SimplePairPotential"]["energy_forces"]["no nlist given"] = @benchmarkable AtomsCalculators.energy_forces($data, $lj)
SUITE["SimplePairPotential"]["energy_forces"]["nlist given"] = @benchmarkable AtomsCalculators.energy_forces($data, $lj; nlist=$nlist)


SUITE["ParametricPairPotential"]["energy"] = BenchmarkGroup()
SUITE["ParametricPairPotential"]["energy_forces"] = BenchmarkGroup()

SUITE["ParametricPairPotential"]["energy"]["no nlist given"] = @benchmarkable AtomsCalculators.potential_energy($data, $ljp)
SUITE["ParametricPairPotential"]["energy"]["nlist given"] = @benchmarkable AtomsCalculators.potential_energy($data, $ljp; nlist=$nlist)

SUITE["ParametricPairPotential"]["energy_forces"]["no nlist given"] = @benchmarkable AtomsCalculators.energy_forces($data, $ljp)
SUITE["ParametricPairPotential"]["energy_forces"]["nlist given"] = @benchmarkable AtomsCalculators.energy_forces($data, $ljp; nlist=$nlist)


SUITE["ParametricPairPotential"]["parameter estimation"] = BenchmarkGroup()

SUITE["ParametricPairPotential"]["parameter estimation"]["energy no nlist given"] = @benchmarkable AtomsCalculators.potential_energy($data, $ljp, $ljp.parameters)
SUITE["ParametricPairPotential"]["parameter estimation"]["energy nlist given"] = @benchmarkable AtomsCalculators.potential_energy($data, $ljp, $ljp.parameters; nlist=$nlist)

SUITE["ParametricPairPotential"]["parameter estimation"]["forces no nlist given"] = @benchmarkable AtomsCalculators.forces($data, $ljp, $ljp.parameters)
SUITE["ParametricPairPotential"]["parameter estimation"]["forces nlist given"] = @benchmarkable AtomsCalculators.forces($data, $ljp, $ljp.parameters; nlist=$nlist)

SUITE["ParametricPairPotential"]["parameter estimation"]["virial no nlist given"] = @benchmarkable AtomsCalculators.virial($data, $ljp, $ljp.parameters)
SUITE["ParametricPairPotential"]["parameter estimation"]["virial nlist given"] = @benchmarkable AtomsCalculators.virial($data, $ljp, $ljp.parameters; nlist=$nlist)