
          
export LennardJones


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
rcut = 6.0u"Å"              
lj = LennardJones(emins, rmins, rcut)
```

It is assumed that the potential is symmetric, i.e. 
`emins[(z1, z2)] == emins[(z2, z1)]` and so forth. 
"""
mutable struct LennardJones{NZ, TZ, T, UL, UE} <: PairPotential 
    zlist::NTuple{NZ, TZ}
    emins::SMatrix{NZ, NZ, T} 
    rmins::SMatrix{NZ, NZ, T}
    rcut::T
end

_fltype(::LennardJones{NZ, TZ, T}) where {NZ, TZ, T} = T 

length_unit(::LennardJones{NZ, TZ, T, UL, UE}) where {NZ, TZ, T, UL, UE} = UL 

energy_unit(::LennardJones{NZ, TZ, T, UL, UE}) where {NZ, TZ, T, UL, UE} = UE 

cutoff_radius(V::LennardJones) = V.rcut * length_unit(V)


function eval_pair(V::LennardJones, r, z1, z2)
    iz1 = _z2i(V.zlist, z1)
    iz2 = _z2i(V.zlist, z2)
    if iz1 == 0 || iz2 == 0 || r > V.rcut
        return zero(r)
    end

    _lj(s) = s^12 - 2 * s^6
    _dlj(s) = 12 * s^11 - 12 * s^5 
    _lj_tilt(s, scut) = _lj(s) - _lj(scut) - (s - scut) * _dlj(scut)

    rmin = V.rmins[iz1, iz2]
    emin = V.emins[iz1, iz2]
    s = rmin / r 
    scut = rmin / V.rcut 

    return emin * _lj_tilt(s, scut)
end


function LennardJones(emins::Dict, rmins::Dict, rcut::Unitful.Length) 

    zlist = tuple( unique(reduce(vcat, collect.(keys(emins))))... )
    NZ = length(zlist)
    TZ = typeof(first(zlist))
    UL = unit(rcut) 
    UE = unit(first(values(emins)))
    T = typeof(ustrip(rcut))
    @assert all(typeof(v) == TZ for v in zlist)
    @assert all(unit(v) == UL for v in values(rmins))
    @assert all(unit(v) == UE for v in values(emins)) 
    @assert all(typeof(ustrip(v)) == T for v in values(rmins))
    @assert all(typeof(ustrip(v)) == T for v in values(emins))
    
    _emin(z1, z2) = haskey(emins, (z1, z2)) ? emins[(z1, z2)] : emins[(z2, z1)]
    _rmin(z1, z2) = haskey(rmins, (z1, z2)) ? rmins[(z1, z2)] : rmins[(z2, z1)]

    emins = SMatrix{NZ, NZ}([ ustrip(_emin(z1, z2)) 
                              for z1 in zlist, z2 in zlist ])
    rmins = SMatrix{NZ, NZ}([ ustrip(_rmin(z1, z2)) 
                              for z1 in zlist, z2 in zlist ])


    return LennardJones{NZ, TZ, T, UL, UE}(zlist, emins, rmins, ustrip(rcut))
end


