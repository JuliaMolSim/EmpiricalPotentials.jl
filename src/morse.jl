
          
export Morse



# ----------- Lennard-Jones potential
# to make this parametric, all we have to do is move the emins and rmins 
# into a `ps` parameter NamedTuple. I suggest to do this in the next iteration 
# and first stabilize the non-parametric potentials. 

""" 
    Morse 

Basic implementation of a (multi-species) Morse potential with finite 
cutoff radius that is imposed by "shifting and tilting" the potential at the 
cutoff. It can be constructed as follows. The parameters are 
  (energy scale, equilibrium bond length, stiffness parameter)
```julia 
params = Dict( (z1, z1) => ( -1.0u"eV",  2.7u"Å", 4.1 ),     
               (z1, z2) => ( -0.5u"eV",  3.2u"Å", 3.5 ),
               (z2, z2) => ( -0.25u"eV", 3.0u"Å", 4.3 ) )
rcut = 6.0u"Å"              
V = Morse(params, rcut) 
```

It is assumed that the potential is symmetric, i.e. 
`params[(z1, z2)] == params[(z2, z1)]`.
"""
mutable struct Morse{NZ, TZ, T, UL, UE} <: PairPotential 
    zlist::NTuple{NZ, TZ}
    params::SMatrix{NZ, NZ, Tuple{T, T, T}}
    rcut::T
end

_fltype(::Morse{NZ, TZ, T}) where {NZ, TZ, T} = T 

length_unit(::Morse{NZ, TZ, T, UL, UE}) where {NZ, TZ, T, UL, UE} = UL 

energy_unit(::Morse{NZ, TZ, T, UL, UE}) where {NZ, TZ, T, UL, UE} = UE 

cutoff_radius(V::Morse) = V.rcut * length_unit(V)


function eval_pair(V::Morse, r, z1, z2)
    iz1 = _z2i(V.zlist, z1)
    iz2 = _z2i(V.zlist, z2)
    if iz1 == 0 || iz2 == 0 || r > V.rcut
        return zero(r)
    end

    _m(s) = exp(-2 * s) - 2 * exp(-s)
    _dm(s) = - 2 * exp(-2 * s) + 2 * exp(-s)
    _m_tilt(s, scut) = _m(s) - _m(scut) - (s - scut) * _dm(scut)

    emin, r0, α = V.params[iz1, iz2]
    s = α * (r / r0 - 1)
    scut = α * (V.rcut / r0 - 1)

    return emin * _m_tilt(s, scut)
end


function Morse(params::Dict, rcut::Unitful.Length)

    zlist = tuple( unique(reduce(vcat, collect.(keys(params))))... )
    NZ = length(zlist)
    TZ = typeof(first(zlist))
    UL = unit(rcut) 
    UE = unit(first(values(params))[1])
    T = typeof(ustrip(rcut))
    @assert all(typeof(v) == TZ for v in zlist)
    @assert all(unit(v[1]) == UE for v in values(params)) 
    @assert all(unit(v[2]) == UL for v in values(params))
    @assert all(typeof(ustrip(v[1])) == T for v in values(params))
    @assert all(typeof(ustrip(v[2])) == T for v in values(params))
    @assert all(typeof(v[3]) == T for v in values(params))
    
    _params(z1, z2) = haskey(params, (z1, z2)) ? params[(z1, z2)] : params[(z2, z1)]

    P = SMatrix{NZ, NZ, Tuple{T, T, T}}(
                    [ustrip.(_params(z1, z2)) for z1 in zlist, z2 in zlist])

    return Morse{NZ, TZ, T, UL, UE}(zlist, P, ustrip(rcut))
end


