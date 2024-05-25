

using DiffResults
using ForwardDiff
using LinearAlgebra: norm
using StaticArrays
using Unitful
import AtomsCalculators

export PairPotential
export ParametricPairPotential
export SimplePairPotential
export LennardJones
using AtomsBase: AbstractSystem 



# NOTE: this could be taken a subtype of SitePotential, but not clear 
#       that this is the best way to do it, one could gain a factor 2 
#       performance by keeping it a separate implementation. 

"""
`PairPotential`:abstractsupertype for pair potentials
"""
abstract type PairPotential <: SitePotential end


_pos_type(sys::AbstractSystem) = typeof(position(sys, 1))

cutoff_radius(pp::PairPotential) = pp.cutoff
Base.zero(pp::PairPotential) = pp.zero_energy

energy_unit(pp::PairPotential) = unit(pp.zero_energy)

length_unit(pp::PairPotential) = unit(pp.cutoff)

AtomsCalculators.zero_forces(sys, calc::PairPotential) = 
        AtomsCalculatorsUtilities.SitePotentials.init_forces(sys, calc)

AtomsCalculators.promote_force_type(sys, calc::PairPotential) = 
        typeof( zero(_pos_type(sys)) * energy_unit(calc)/length_unit(calc) )

##

struct SimplePairPotential{ID, TC, TE} <: PairPotential where {TC<:Unitful.LengthUnits, TE<:Unitful.EnergyUnits }
    f::Function
    atom_ids::NTuple{2,ID}
    cutoff::TC
    "zero used to define output type and unit"
    zero_energy::TE  
end


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
            te::Float64 = DiffResults.value(d_result)  # type instablity here
            e_tmp += te
            tmp::Float64 =  DiffResults.derivative(d_result)  # type instablity here
            f[i] = ( tmp / (2r) ) * R  # divide with two here to take off double count
        end
    end
    return e_tmp/2, f
end

##

struct ParametricPairPotential{ID, TP, TC, TE} <: PairPotential where {TP<:AbstractArray, TC<:Unitful.LengthUnits, TE<:Unitful.EnergyUnits }
    f::Function
    parameters::TP
    atom_ids::NTuple{2,ID}
    cutoff::TC
    "zero used to define output type and unit"
    zero_energy::TE  
end


function eval_site(ppp::ParametricPairPotential, Rs, Zs, z0)
    if ! (z0 in ppp.atom_ids)
        return ustrip(zero(ppp))
    end
    tmp = ustrip(zero(ppp))
    id = z0 == ppp.atom_ids[1] ?  ppp.atom_ids[2] : ppp.atom_ids[1]
    for (i, R) in zip(Zs, Rs)
        if i == id
            r = norm(R)
            tmp += ppp.f(r, ppp.parameters)
        end
    end
    return tmp/2 # divide by to to get correct double count
end


function eval_grad_site(ppp::ParametricPairPotential, Rs, Zs, z0)
    @assert length(Rs) == length(Zs)
    T = promote_type( (typeof ∘ ustrip ∘ zero)(ppp) , (eltype ∘ eltype)(Rs) )
    f = zeros(SVector{3, T}, length(Zs))
    e_tmp = ustrip(ppp.zero_energy)
    if ! (z0 in ppp.atom_ids)  # potential is not defined for this case
        return e_tmp, f  # return zeros - this is not the optimal but will do for now
    end
    id = z0 == ppp.atom_ids[1] ?  ppp.atom_ids[2] : ppp.atom_ids[1]
    d_result = DiffResults.DiffResult(zero(T), zero(T))
    for (i, Z, R) in zip(1:length(Zs), Zs, Rs)
        if Z == id
            r = norm(R)
            d_result = ForwardDiff.derivative!(d_result, x->ppp.f(x, ppp.parameters), r)
            te::T= DiffResults.value(d_result)  # type instablity here
            e_tmp += te
            tmp::T =  DiffResults.derivative(d_result)  # type instablity here
            f[i] = ( tmp / (2r) ) * R  # divide with two here to take off double count
        end
    end
    return e_tmp/2, f
end


# Parameter estimation ∂f/∂params
function eval_site(ppp::ParametricPairPotential, params::AbstractArray, Rs, Zs, z0)
    T = promote_type( (typeof ∘ ustrip ∘ zero)(ppp) , (eltype ∘ eltype)(Rs) )
    tmp = zeros( T, length(params) )
    if ! (z0 in ppp.atom_ids)
        return tmp
    end
    id = z0 == ppp.atom_ids[1] ?  ppp.atom_ids[2] : ppp.atom_ids[1]
    for (i, R) in zip(Zs, Rs)
        if i == id
            r = norm(R)
            tmp += ForwardDiff.gradient( a -> ppp.f(r, a), params)
        end
    end
    return tmp/2 # divide by to to get correct double count
end


function eval_grad_site(ppp::ParametricPairPotential, params::AbstractArray, Rs, Zs, z0)
    @assert length(Rs) == length(Zs)
    # Plan is to calculate ∂E/∂rᵢ∂pⱼ with Hessian calculation
    # using [r, params...] as imput.
    T = promote_type( (typeof ∘ ustrip ∘ zero)(ppp) , (eltype ∘ eltype)(Rs) )
    m = SMatrix{length(Rs[1]), length(params)}(zeros(T, length(Rs[1]), length(params)))
    f = fill( m, length(Zs) )
    if ! (z0 in ppp.atom_ids)  # potential is not defined for this case
        return f  # return zeros - this is not the optimal but will do for now
    end
    id = z0 == ppp.atom_ids[1] ?  ppp.atom_ids[2] : ppp.atom_ids[1]

    for (i, Z, R) in zip(1:length(Zs), Zs, Rs)
        if Z == id
            r = norm(R)
            hess = ForwardDiff.hessian( a -> ppp.f(a[1], a[2:end]), [r, params...] )
            f[i] = [ ( tmp / (2r) ) * Rᵢ for Rᵢ in R, tmp in @view hess[2:end, 1] ]
        end
    end
    return f
end


##


function LennardJones(
    emin::Unitful.Energy,
    rmin::Unitful.Length,
    id1, id2,
    cutoff::Unitful.Length;
    parametric=false
)
    @assert emin < 0u"eV"
    @assert rmin > 0u"pm"
    @assert cutoff > rmin
    ε = -ustrip(emin)
    σ = ustrip(unit(cutoff), rmin) / 2^(1//6)
    A = 4ε * σ^12
    B = 4ε * σ^6

    if parametric
        return ParametricPairPotential(
            (r,c) -> c[1]/r^12 - c[2]/r^6,
            SVector(A, B),
            (id1, id2),
            cutoff,
            zero(emin)
        )
    else
        return SimplePairPotential(
            r -> A/r^12 - B/r^6,
            (id1, id2),
            cutoff,
            zero(emin)
        )
    end
end
