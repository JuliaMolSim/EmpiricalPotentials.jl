
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


function _ad_sw_bondangle_(R)
   R1, R2 = R[1:3], R[4:6]
   r1, r2 = norm(R1), norm(R2)
   return sw_bondangle(R1/r1, R2/r2)
end

# TODO: need a faster implementation of sw_bondangle_dd
function sw_bondangle_dd(R1, R2)
   R = [R1; R2]
   hh = ForwardDiff.hessian(_ad_sw_bondangle_, R)
   h = zeros(JMatF, 2,2)
   h[1,1] = JMatF(hh[1:3,1:3])
   h[1,2] = JMatF(hh[1:3, 4:6])
   h[2,1] = JMatF(hh[4:6,1:3])
   h[2,2] = JMatF(hh[4:6, 4:6])
   return h
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


function eval_grad_site(calc::StillingerWeber, Rs, Zs, z0)
   # TODO: insert assertion that all Zs are Si and z0 is also Si
   f = zeros(SVector{3, Float64}, length(Zs))
   Eᵢ = ustrip(zero(calc)) 
   if z0 != calc.atomic_id
      return Eᵢ, f
   end
   Rs = [ rᵢ for (zᵢ, rᵢ) in zip(Zs,Rs) if zᵢ == calc.atomic_id ]
   ind = [ i for (i, zᵢ) in enumerate(Zs) if zᵢ == calc.atomic_id ]
   Zs = filter(id-> id == calc.atomic_id, Zs)

   TF = eltype(eltype(Rs))
   Nr = length(Rs)
   r = acquire!(calc.pool, :r, (Nr,), TF)
   S = acquire!(calc.pool, :S, (Nr,), SVector{3, TF})
   V3 = acquire!(calc.pool, :V3, (Nr,), TF)
   gV3 = acquire!(calc.pool, :gV3, (Nr,), SVector{3, TF})
   dEs = acquire!(calc.pool, :dEs, (Nr,), SVector{3, TF})

   d_result = DiffResults.DiffResult(Eᵢ, Eᵢ)
   for j in eachindex(Rs)
      r[j] = rⱼ = norm(Rs[j])
      S[j] = 𝐫̂ⱼ = Rs[j] / rⱼ
      
      d_result = ForwardDiff.derivative!(d_result, calc.V3, rⱼ)
      te::TF = DiffResults.value(d_result)
      V3[j] = te
      tmp::TF =  DiffResults.derivative(d_result)
      gV3[j] = tmp * 𝐫̂ⱼ

      d_result = ForwardDiff.derivative!(d_result, calc.V2, rⱼ)
      te = DiffResults.value(d_result)
      Eᵢ += te
      tmp =  DiffResults.derivative(d_result)
      dEs[j] = tmp * (𝐫̂ⱼ/2)
   end

   for j₁ in 1:Nr, j₂ in j₁+1:Nr
      Eᵢ += V3[j₁] * V3[j₂] * sw_bondangle(S[j₁], S[j₂])
      a, b₁, b₂ = sw_bondangle_d(S[j₁], S[j₂], r[j₁], r[j₂])
      dEs[j₁] += (V3[j₁] * V3[j₂]) * b₁ + (V3[j₂] * a) * gV3[j₁]
      dEs[j₂] += (V3[j₁] * V3[j₂]) * b₂ + (V3[j₁] * a) * gV3[j₂]
   end
   
   # Index conversion back
   for (i,j)  in enumerate(ind)
      f[j] = dEs[i]
   end

   release!(r); release!(S); release!(V3); release!(gV3); release!(dEs)

   return Eᵢ, f
end



# This code seems to have a bug and is therefore removed for now
# function hess(V::StillingerWeber, r, R)
#    n = length(r)
#    hV = zeros(JMatF, n, n)
#
#    # two-body contributions
#    for (i, (r_i, R_i)) in enumerate(zip(r, R))
#       hV[i,i] += hess(V.V2, r_i, R_i)
#    end
#
#    # three-body terms
#    S = [ R1/r1 for (R1,r1) in zip(R, r) ]
#    V3 = [ V.V3(r1) for r1 in r ]
#    dV3 = [ grad(V.V3, r1, R1) for (r1, R1) in zip(r, R) ]
#    hV3 = [ hess(V.V3, r1, R1) for (r1, R1) in zip(r, R) ]
#
#    for i1 = 1:(length(r)-1), i2 = (i1+1):length(r)
#       # Es += V3[i1] * V3[i2] * bondangle(S[i1], S[i2])
#       # precompute quantities
#       ψ, Dψ_i1, Dψ_i2 = bondangle_d(S[i1], S[i2], r[i1], r[i2])
#       Hψ = bondangle_dd(R[i1], R[i2])  # <<<< this should be SLOW (AD)
#       # assemble local hessian contributions
#       hV[i1,i1] +=
#          hV3[i1] * V3[i2] * ψ       +   dV3[i1] * V3[i2] * Dψ_i1' +
#          Dψ_i1 * V3[i2] * dV3[i1]'  +   V3[i1] * V3[i2] * Hψ[1,1]
#       hV[i2,i2] +=
#          V3[i2] * hV3[i2] * ψ       +   V3[i1] * dV3[i2] * Dψ_i2' +
#          Dψ_i2 * V3[i1] * dV3[i2]'  +   V3[i1] * V3[i2] * Hψ[2,2]
#       hV[i1,i2] +=
#          dV3[i1] * dV3[i2]' * ψ     +   V3[i1] * Dψ_i1 * dV3[i2]' +
#          dV3[i1] * V3[i2] * Dψ_i2'  +   V3[i1] * V3[i2] * Hψ[1,2]
#       hV[i2, i1] +=
#          dV3[i2] * dV3[i1]' * ψ     +   V3[i1] * Dψ_i2 * dV3[i1]' +
#          dV3[i2] * V3[i1] * Dψ_i1'  +   V3[i1] * V3[i2] * Hψ[2,1]
#    end
#    return hV
# end


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