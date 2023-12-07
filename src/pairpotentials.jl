using DiffResults
using ForwardDiff
using LinearAlgebra: norm
using StaticArrays
using Unitful


export PairPotential
export SimplePairPotential


# NOTE: this could be taken a subtype of SitePotential, but not clear 
#       that this is the best way to do it, one could gain a factor 2 
#       performance by keeping it a separate implementation. 

"""
`PairPotential`:abstractsupertype for pair potentials
"""
abstract type PairPotential <: SitePotential end




struct SimplePairPotential{ID, TC, TE} <: PairPotential where {TC<:Unitful.LengthUnits, TE<:Unitful.EnergyUnits }
    f::Function
    atom_ids::NTuple{2,ID}
    cutoff::TC
    "zero used to define output type and unit"
    zero_energy::TE  
end

cutoff_radius(spp::SimplePairPotential) = spp.cutoff
Base.zero(spp::SimplePairPotential) = spp.zero_energy


function eval_site(spp::SimplePairPotential, Rs, Zs, z0)
    if ! (z0 in spp.atom_ids)
        return ustrip(zero(spp))
    end
    tmp = ustrip(zero(spp))
    id = z0 == spp.atom_ids[1] ?  spp.atom_ids[2] : spp.atom_ids[1]
    for (i, R) in zip(Zs, Rs)
        if i == id
            r = norm(R)
            tmp += spp.f(r)
        end
    end
    return tmp/2 # divide by to to get correct double count
end


function eval_grad_site(spp::SimplePairPotential, Rs, Zs, z0)
    @assert length(Rs) == length(Zs)
    f = zeros(SVector{3, Float64}, length(Zs))
    e_tmp = ustrip(spp.zero_energy)
    if ! (z0 in spp.atom_ids)  # potential is not defined for this case
        return e_tmp, f  # return zeros - this is not the optimal but will do for now
    end
    id = z0 == spp.atom_ids[1] ?  spp.atom_ids[2] : spp.atom_ids[1]
    d_result = DiffResults.DiffResult(e_tmp, e_tmp)
    for (i, Z, R) in zip(1:length(Zs), Zs, Rs)
        if Z == id
            r = norm(R)
            d_result = ForwardDiff.derivative!(d_result, spp.f, r)
            e_tmp += DiffResults.value(d_result)
            f[i] = ( DiffResults.derivative(d_result) / (2r) ) * R  # divide with two here to take off double count
        end
    end
    return e_tmp/2, f
end