
import AtomsCalculatorsUtilities.PairPotentials: PairPotential, 
            eval_pair, cutoff_radius 

import AtomsCalculators: energy_unit, length_unit 

using StaticArrays
          
export LennardJones

#, Morse, ZBL 


# a simple utility to find a species index in a short list 

function _z2i(zlist, z)
    for iz = 1:length(zlist)
        if zlist[iz] == z
            return iz
        end
    end
    return 0 
end 


# ----------- Lennard-Jones potential
# to make this parametric, all we have to do is move the emins and rmins 
# into a `ps` parameter NamedTuple. I suggest to do this in the next iteration 
# and first stabilize the non-parametric potentials. 

""" 
    LennardJones 

Basic implementation of a (multi-species) Lennard-Jones potential with finite 
cutoff radius that is imposed by "shifting and tilting" the potential at the 
cutoff. It can be constructed as follows.
```julia 
emins = Dict( (z1, z1) => -1.0u"eV", 
              (z1, z2) => -0.5u"eV", 
              (z2, z2) => -0.25u"eV" )
rmins = Dict( (z1, z1) => 2.7u"Å", 
              (z1, z2) => 3.2u"Å", 
              (z2, z2) => 3.0u"Å" )
cutoff = 6.0u"Å"              
lj = LennardJones(emins, rmins, cutoff)
```

It is assumed that the potential is symmetric, i.e. 
`emins[(z1, z2)] == emins[(z2, z1)]` and so forth. 
"""
struct LennardJones{NZ, TZ, T, UL, UE} <: PairPotential 
    zlist::NTuple{NZ, TZ}
    emins::SMatrix{NZ, NZ, T} 
    rmins::SMatrix{NZ, NZ, T}
    cutoff::T
end

_fltype(::LennardJones{NZ, TZ, T}) where {NZ, TZ, T} = T 

length_unit(::LennardJones{NZ, TZ, T, UL, UE}) where {NZ, TZ, T, UL, UE} = UL 

energy_unit(::LennardJones{NZ, TZ, T, UL, UE}) where {NZ, TZ, T, UL, UE} = UE 

cutoff_radius(V::LennardJones) = V.cutoff * length_unit(V)


function eval_pair(V::LennardJones, r, z1, z2)
    iz1 = _z2i(V.zlist, z1)
    iz2 = _z2i(V.zlist, z2)
    if iz1 == 0 || iz1 == 0 
        return zero(_fltype(V))
    end

    _lj(s) = s^12 - 2 * s^6
    _dlj(s) = 12 * s^11 - 12 * s^5 
    _lj_tilt(s, scut) = _lj(s) - _lj(scut) - (s - scut) * _dlj(scut)

    rmin = V.rmins[iz1, iz2]
    emin = V.emins[iz1, iz2]
    s = rmin / r 
    scut = rmin / V.rcut 

    return emin * _lj_tilt(s)
end


function LennardJones(emins::Dict, rmins::Dict, cutoff::Unitful.Length) 

    zlist = unique(reduce(vcat, collect.(keys(emins))))
    NZ = length(zlist)

    _emin(z1, z2) = haskey(emins, (z1, z2)) ? emins[(z1, z2)] : emins[(z2, z1)]
    _rmin(z1, z2) = haskey(rmins, (z1, z2)) ? rmins[(z1, z2)] : rmins[(z2, z1)]

    emins = SMatrix{NZ, NZ}([ _emin(z1, z2) for z1 in zlist, z2 in zlist ])
    rmins = SMatrix{NZ, NZ}([ _rmin(z1, z2) for z1 in zlist, z2 in zlist ])

    return LennardJones(zlist, emins, rmins, cutoff)
end
