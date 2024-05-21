
# NOTE: this is a quick and dirty implementation of the Stillinger-Weber model;
# in the future it will be good to do this more elegantly / more efficiently
# there is huge overhead in this code that can be
#   significantly improved in terms of performance, but for now we just
#   want a correct and readable code

# [Stillinger/Weber, PRB 1985]
# ---------------------------------------------------
# v2 = ϵ f2(r_ij / σ); f2(r) = A (B r^{-p} - r^{-q}) exp( (r-a)^{-1} )
# v3 = ϵ f3(ri/σ, rj/σ, rk/σ); f3 = h(rij, rij, Θjik) + ... + ...
# h(rij, rik, Θjik) = λ exp[ γ (rij-a)^{-1} + γ (rik-a)^{-1} ] * (cosΘjik+1/3)^2
#       >>>
# V2 = 0.5 * ϵ * A (B r^{-p} - r^{-q}) * exp( (r-a)^{-1} )
# V3 = √ϵ * λ exp[ γ (r-a)^{-1} ]
#
# Parameters from QUIP database:
# -------------------------------
# <per_pair_data atnum_i="14" atnum_j="14" AA="7.049556277" BB="0.6022245584"
         # p="4" q="0" a="1.80" sigma="2.0951" eps="2.1675" />
# <per_triplet_data atnum_c="14" atnum_j="14" atnum_k="14" lambda="21.0"
         # gamma="1.20" eps="2.1675" />


using ForwardDiff, ObjectPools, StaticArrays
using ForwardDiff: Dual 

using LinearAlgebra: dot, norm 

export StillingerWeber



"""
`sw_bondangle(S1, S2) -> (dot(S1, S2) + 1.0/3.0)^2`

* not this assumes that `S1, S2` are normalised
* see `sw_bondangle_d` for the derivative
"""
sw_bondangle(S1, S2) = (dot(S1, S2) + 1.0/3.0)^2

"""
`b := sw_bondangle(S1, S2)` then

`sw_bondangle_d(S1, S2, r1, r2) -> b, db1, db2`

where `dbi` is the derivative of `b` w.r.t. `Ri` where `Si= Ri/ri`.
"""
function sw_bondangle_d(S1, S2, r1, r2)
   d = dot(S1, S2)
   b1 = (1.0/r1) * S2 - (d/r1) * S1
   b2 = (1.0/r2) * S1 - (d/r2) * S2
   return (d+1.0/3.0)^2, 2.0*(d+1.0/3.0)*b1, 2.0*(d+1.0/3.0)*b2
end



"""
Stillinger-Weber potential with parameters for Si.

Functional form and default parameters match the original SW potential
from [Stillinger/Weber, PRB 1985].

The `StillingerWeber` type can also by "abused" to generate arbitrary
bond-angle potentials of the form
   Σᵢⱼ V₂(rᵢⱼ) + Σᵢⱼₖ V₃(rᵢⱼ) V₃(rᵢₖ) (cos Θᵢⱼₖ + 1/3)²

Constructor admits the following key-word parameters:
`ϵ=2.1675, σ = 2.0951, A=7.049556277, B=0.6022245584,
   p = 4, a = 1.8, λ=21.0, γ=1.20, atom_number=14`

which enter the potential as follows:
```
V2(r) = 0.5 * ϵ * A * (B * (r/σ)^(-p) - 1.0) * exp(1.0 / (r/σ - a))
V3(r) = sqrt(ϵ * λ) * exp(γ / (r/σ - a))
```

The `brittle` keyword can be used to switch to the parameters λ to 42.0, 
which is appropriate to simulate brittle fracture. (need reference for this)
"""
struct StillingerWeber{P1, P2, T, TI} <: SitePotential
   V2::P1
   V3::P2
   rcut::T
   pool::TSafe{ArrayPool{FlexArrayCache}}
   meta::Dict{Symbol, T}
   atomic_id::TI
end

cutoff_radius(calc::StillingerWeber) = calc.rcut * u"Å"

function StillingerWeber(;
   brittle = false,
   ϵ = 2.1675, 
   σ = 2.0951, 
   A = 7.049556277, 
   B = 0.6022245584,
   p = 4, 
   a = 1.8, 
   λ = brittle ? 42.0 : 21.0, 
   γ = 1.20,
   atom_number = 14  # = Si
)
   rcut = a * σ - 1e-2 # the 1e-2 is due to avoid the numerical issues 
                       # in the exp(1.0/(r/σ - a) like terms 
   V2 = r -> r >= rcut ? 0.0 : (ϵ*A) * (B*(r/σ)^(-p) - 1.0) * exp(1.0/(r/σ - a))
   V3 = r -> r >= rcut ? 0.0 : sqrt(ϵ * λ) * exp( γ / (r/σ - a) )
   meta = Dict(:ϵ => ϵ, :σ => σ, :A => A, :B => B, :p => p, :a => a,
               :λ => λ, :γ => γ, :brittle => brittle, :rcut => rcut)
   return StillingerWeber(
      V2, 
      V3, 
      rcut, 
      TSafe(ArrayPool(FlexArrayCache)), 
      meta,
      atom_number
   ) 
end


function eval_site(calc::StillingerWeber, Rs, Zs, z0)
   if z0 != calc.atomic_id
      return ustrip(zero(calc))
   end
   Rs = [ rᵢ for (zᵢ, rᵢ) in zip(Zs,Rs) if zᵢ == calc.atomic_id ]
   Zs = filter(id-> id == calc.atomic_id, Zs)

   TF = eltype(eltype(Rs))
   Nr = length(Rs)
   Eᵢ = zero(TF) 
   S = acquire!(calc.pool, :S, (Nr,), SVector{3, TF})
   V3 = acquire!(calc.pool, :V3, (Nr,), TF)

   for j in eachindex(Rs)
      r = norm(Rs[j])
      S[j] = Rs[j] / r  # S[j] is used later
      V3[j] = calc.V3(r)
      Eᵢ += calc.V2(r) / 2
   end

   for j₁ in 1:Nr, j₂ in j₁+1:Nr
      Eᵢ += V3[j₁] * V3[j₂] * sw_bondangle(S[j₁], S[j₂])
   end

   release!(S); release!(V3)

   return Eᵢ
end


# function eval_grad_site(calc::StillingerWeber, Rs, Zs, z0)
#    TF = eltype(eltype(Rs))
#    f = zeros(SVector{3, TF}, length(Zs))
#    Eᵢ = zero(TF)
#    # if z0 != calc.atomic_id
#    #    return Eᵢ, f
#    # end
#    # Rs = [ rᵢ for (zᵢ, rᵢ) in zip(Zs,Rs) if zᵢ == calc.atomic_id ]
#    # ind = [ i for (i, zᵢ) in enumerate(Zs) if zᵢ == calc.atomic_id ]
#    # Zs = filter(id-> id == calc.atomic_id, Zs)
#    @assert z0 == calc.atomic_id
#    @assert all(Zs .== calc.atomic_id)

#    Nr = length(Rs)
#    r = acquire!(calc.pool, :r, (Nr,), TF)
#    S = acquire!(calc.pool, :S, (Nr,), SVector{3, TF})
#    V3 = acquire!(calc.pool, :V3, (Nr,), TF)
#    gV3 = acquire!(calc.pool, :gV3, (Nr,), SVector{3, TF})
#    dEs = acquire!(calc.pool, :dEs, (Nr,), SVector{3, TF})

#    d_result = DiffResults.DiffResult(Eᵢ, Eᵢ)
#    for j in eachindex(Rs)
#       r[j] = rⱼ = norm(Rs[j])
#       S[j] = 𝐫̂ⱼ = Rs[j] / rⱼ
      
#       d_result = ForwardDiff.derivative!(d_result, calc.V3, rⱼ)
#       te::TF = DiffResults.value(d_result)
#       V3[j] = te
#       tmp::TF =  DiffResults.derivative(d_result)
#       gV3[j] = tmp * 𝐫̂ⱼ

#       d_result = ForwardDiff.derivative!(d_result, calc.V2, rⱼ)
#       te = DiffResults.value(d_result)
#       Eᵢ += te
#       tmp =  DiffResults.derivative(d_result)
#       dEs[j] = tmp * (𝐫̂ⱼ/2)
#    end

#    for j₁ in 1:Nr, j₂ in j₁+1:Nr
#       Eᵢ += V3[j₁] * V3[j₂] * sw_bondangle(S[j₁], S[j₂])
#       a, b₁, b₂ = sw_bondangle_d(S[j₁], S[j₂], r[j₁], r[j₂])
#       dEs[j₁] += (V3[j₁] * V3[j₂]) * b₁ + (V3[j₂] * a) * gV3[j₁]
#       dEs[j₂] += (V3[j₁] * V3[j₂]) * b₂ + (V3[j₁] * a) * gV3[j₂]
#    end
   
#    # Index conversion back
#    # for (i,j)  in enumerate(ind)
#    for j = 1:Nr
#       f[j] = dEs[j]
#    end

#    release!(r); release!(S); release!(V3); release!(gV3); release!(dEs)

#    return Eᵢ, f
# end


function eval_grad_site(calc::StillingerWeber, Rs, Zs, z0)
   Nr = length(Rs)
   @assert length(Zs) == Nr
   @assert z0 == calc.atomic_id
   @assert all(Zs .== calc.atomic_id)

   TF = eltype(eltype(Rs))
   Ei = zero(TF)
   dEi = zeros(SVector{3, TF}, length(Zs))

   r = acquire!(calc.pool, :r, (Nr,), TF)
   S = acquire!(calc.pool, :S, (Nr,), SVector{3, TF})
   V3 = acquire!(calc.pool, :V3, (Nr,), TF)
   gV3 = acquire!(calc.pool, :gV3, (Nr,), SVector{3, TF})

   for j = 1:Nr 
      r[j] = rⱼ = norm(Rs[j])
      S[j] = 𝐫̂ⱼ = Rs[j] / rⱼ
      
      dv3 = calc.V3(Dual(rⱼ, one(rⱼ)))
      V3[j] = ForwardDiff.value(dv3) 
      gV3[j] = ForwardDiff.partials(dv3, 1) * 𝐫̂ⱼ/2

      dv2 = calc.V2(Dual(rⱼ, one(rⱼ)))
      Ei += ForwardDiff.value(dv2)
      dEi[j] = ForwardDiff.partials(dv2, 1) * 𝐫̂ⱼ/2
   end

   for j₁ in 1:Nr, j₂ in j₁+1:Nr
      Ei += V3[j₁] * V3[j₂] * sw_bondangle(S[j₁], S[j₂])
      a, b₁, b₂ = sw_bondangle_d(S[j₁], S[j₂], r[j₁], r[j₂])
      dEi[j₁] += (V3[j₁] * V3[j₂]) * b₁ + (V3[j₂] * a) * gV3[j₁]
      dEi[j₂] += (V3[j₁] * V3[j₂]) * b₂ + (V3[j₁] * a) * gV3[j₂]
   end
   
   release!(r); release!(S); release!(V3); release!(gV3);

   return Ei, dEi
end


# ---------------------------------------------------
# the hessian implementation just uses the ForwardDiff fallbacks 

block_hessian_site(sw::StillingerWeber, args...) = 
         ad_block_hessian_site(sw, args...)


hessian_site(sw::StillingerWeber, args...) = 
         ad_hessian_site(sw, args...)


# ---------------------------------------------------
#  the preconditioner implementation


# ∇V  = V'(r) 𝐫̂ 
# ∇²V = V''(r) 𝐫̂ ⊗ 𝐫̂ - (V'(r)/r) (I - 𝐫̂ ⊗ 𝐫̂)

using LinearAlgebra: I, mul! 

function _precon_pair(V, r::T, 𝐫̂::SVector{3, T}) where {T} 
   dVfun  = _r -> ForwardDiff.derivative(V,     _r)
   ddVfun = _r -> ForwardDiff.derivative(dVfun, _r) 
   dv = dVfun(r) 
   ddv = ddVfun(r) 
   P = 𝐫̂ * 𝐫̂'
   Pperp = SMatrix{3,3,T}(I) - P 
   return abs(ddv) * P + abs(dv/r) * Pperp
end

function precon(calc::StillingerWeber, Rs::AbstractVector{SVector{3, TF}}, 
                Zs, z0, innerstab=0.1) where {TF}
   Nr = length(Rs)
   r = acquire!(calc.pool, :r, (Nr,), TF)
   S = acquire!(calc.pool, :S, (Nr,), SVector{3, TF})
   V3 = acquire!(calc.pool, :V3, (Nr,), TF)

   I3x3 = SMatrix{3, 3, TF}(I)
   Z3x3 = zero(SMatrix{3, 3, TF})
   _pEs = acquire!(calc.pool, :pEs, (Nr, Nr), typeof(I3x3))
   pEs = unwrap(_pEs)
   for i = 1:Nr, j = 1:Nr 
      pEs[i, j] = Z3x3 
   end 

   # two-body contributions
   for (i, 𝐫) in enumerate(Rs)
      r[i] = ri = norm(𝐫)
      V3[i] = calc.V3(ri)
      S[i] = 𝐫̂i = 𝐫 / ri
      pEs[i,i] += ( (1 - innerstab) * _precon_pair(calc.V2, ri, 𝐫̂i) 
                     + innerstab * I3x3) 
   end

   # three-body terms
   for i1 = 1:(Nr-1), i2 = (i1+1):Nr
      Θ = dot(S[i1], S[i2])
      dΘ1 = (one(TF)/r[i1]) * S[i2] - (Θ/r[i1]) * S[i1]
      dΘ2 = (one(TF)/r[i2]) * S[i1] - (Θ/r[i2]) * S[i2]
      # ψ = (Θ + 1/3)^2, ψ' = 2 (Θ + 1/3), ψ'' = 2
      a = (1 - innerstab) * abs(V3[i1] * V3[i2] * 2)
      pEs[i1, i2] += a * dΘ1 * dΘ2'
      pEs[i1, i1] += a * dΘ1 * dΘ1'
      pEs[i2, i2] += a * dΘ2 * dΘ2'
      pEs[i2, i1] += a * dΘ2 * dΘ1'
   end

   return _pEs
end