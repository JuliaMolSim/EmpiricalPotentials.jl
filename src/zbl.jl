

# @pot ZBLPotential

# ZBLPotential() = (
#    let
#       # au = 0.8854 * 0.529 / (Z1^0.23 + Z2^0.23)
#       ϵ0 = 0.00552634940621
#       C = 1/(4*π*ϵ0)
#       E1, E2, E3, E4 = 0.1818, 0.5099, 0.2802, 0.02817
#       A1, A2, A3, A4 = 3.2, 0.9423, 0.4028, 0.2016
#       V = @analytic(r -> C * (E1*exp(-A1*r) + E2*exp(-A2*r) +
#                               E3*exp(-A4*r) + E4*exp(-A4*r) ) / r)
#       ZBLPotential(V)
#    end)

# _zbl_au(Z1, Z2) = (0.8854 * 0.529) / (Z1^0.23 + Z2^0.23)

# function evaluate(V::ZBLPotential, r::Number, z1, z0)
#    au = _zbl_au(z1, z0)
#    return evaluate(V.V, r / au)
# end

export ZBL 


""" 
    ZBL 

Basic implementation of a ZBL potential (TODO insert reference). The original 
ZBL has not cutoff but the interface we use enforces a cutoff. This is the only 
parameter. It can be constructed as follows.
```julia 
rcut = 6.0u"Å"              
zbl = ZBL(rcut) 
```

The current version of this potential assumes eV and Å as energy and force 
units. A PR to generalize this is welcome. 
"""
mutable struct ZBL{T} <: PairPotential 
    rcut::T
    fcut::T 
    dfcut::T 
end

_fltype(::ZBL{T}) where {T} = T 

length_unit(::ZBL) = u"Å"

energy_unit(::ZBL) = u"eV"

cutoff_radius(V::ZBL) = V.rcut * length_unit(V)


_zbl_au(Z1, Z2) = (0.8854 * 0.529) / (Z1^0.23 + Z2^0.23)

function _f_zbl(r) 
    ϵ0 = 0.00552634940621
    C = 1/(4*π*ϵ0)
    E1, E2, E3, E4 = 0.1818, 0.5099, 0.2802, 0.02817
    A1, A2, A3, A4 = 3.2, 0.9423, 0.4028, 0.2016
    return C * (E1*exp(-A1*r) + E2*exp(-A2*r) +
                    E3*exp(-A4*r) + E4*exp(-A4*r) ) / r
end


function eval_pair(V::ZBL, r, z1, z2)
    if r > V.rcut
        return zero(typeof(r)) 
    end

    return _zbl_au(z1, z2) * (_f_zbl(r) - V.fcut - (r - V.rcut) * V.dfcut)
end


function ZBL(rcut::Unitful.Length)
    @assert unit(rcut) == u"Å"
    _rcut = ustrip(rcut)
    fcut = _f_zbl(_rcut)
    dfcut = ForwardDiff.derivative(_f_zbl, _rcut)
    return ZBL(_rcut, fcut, dfcut)
end 

    
