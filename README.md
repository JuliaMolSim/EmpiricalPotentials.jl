# EmpiricalPotentials

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaMolSim.github.io/EmpiricalPotentials.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaMolSim.github.io/EmpiricalPotentials.jl/dev/)
[![Build Status](https://github.com/JuliaMolSim/EmpiricalPotentials.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaMolSim/EmpiricalPotentials.jl/actions/workflows/CI.yml?query=branch%3Amain)

Implementation of `AtomsBase` and `AtomsCalculators` compatible 
empirical interatomic potentials. At the moment, the following 
potentials are provided: 
- LennardJones (multi-species) 
- Morse (multi-species)
- ZBL
- StillingerWeber (Si)

EAM is planned, but there is no ETA. At the moment potentials have fixed parameters. Extension for parameterized potentials (low-level AtomsCalculators interface) are planned, but also no ETA. 

Issues to request adding other potentials or functionality or PRs implementing them are welcome.