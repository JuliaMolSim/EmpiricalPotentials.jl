
using AtomsBase
using AtomsCalculators
using ChunkSplitters
using Folds
using NeighbourLists
using ObjectPools: acquire!, release!
using Unitful 


"""
`SitePotential`:abstractsupertype for generic site potentials. Concrete subtypes 
should overload the `cutoff_radius`, `eval_site` and `eval_grad_site` methods. 
"""
abstract type SitePotential end

"""
write docs...
"""
function cutoff_radius end 

"""
If `V <: SitePotential` then it should implement the method
```julia 
eval_site(V, Rs, Zs, z0)
```
where `Rs::AbstractVector{<: SVector{3}}` and `Zs::AbstractVector` of atomic 
numbers, while `z0` is a single `AtomicNumber`.
"""
function eval_site end


"""
If `V <: SitePotential` then it should implement the method
```julia 
eval_grad_site(V, Rs, Zs, z0)
```
where `Rs::AbstractVector{<: SVector{3}}` and `Zs::AbstractVector` of atomic 
numbers, while `z0` is a single `AtomicNumber`.
"""
function eval_grad_site end 



# Used to define ouput type and unit.
# Needed to get empty domain working
Base.zero(::SitePotential) = zero(1.0u"eV")

# These two are also needed but should be ok with these
energy_unit(V) = unit(zero(V))
force_unit(V) = energy_unit(V) / unit( cutoff_radius(V) )

# No need to change this
AtomsCalculators.promote_force_type(::Any, spp::SitePotential) = SVector{3, typeof(zero(spp)/cutoff_radius(spp))}

# Tune this for your potentials id type
# default is atomic number, but you could use atomic symbol or something else
@inline get_id(at, V, i) = atomic_number(at, i)

"""
`get_neighbours(nlist::PairList, at, i::Integer) -> Js, Rs, Zs, z0`
"""
function get_neighbours(at, V, nlist, i)
   Js, Rs = NeighbourLists.neigs(nlist, i)
   Zs = [ get_id(at, V, j) for j in Js ]
   z0 = get_id(at, V, i) 
   return Js, Rs, Zs, z0 
end


AtomsCalculators.@generate_interface function AtomsCalculators.potential_energy(
   at, 
   V::SitePotential; 
   domain=1:length(at), 
   executor=ThreadedEx(), 
   nlist=nothing, 
   kwargs...
) 
   if isnothing(nlist)
      nlist = PairList(at, cutoff_radius(V))
   end

   E = Folds.sum( domain, executor; init=ustrip(zero(V)) ) do i
      Js, Rs, Zs, z0 = get_neighbours(at, V, nlist, i) 
      e_i = eval_site(V, Rs, Zs, z0)
      release!(Js); release!(Rs); release!(Zs)
      e_i
   end

   return E * energy_unit(V)
end


function site_energy(
   at, 
   V::SitePotential; 
   domain=1:length(at), 
   executor=ThreadedEx(), 
   nlist=nothing, 
   kwargs...
) 
   # this has a problem: if domain is empty, then `sum` needs an initializer 
   # and I don't know how to infer the type of the output. I JuLIP this was 
   # solved via `fltype(V)` or something like that ... 

   # function _site_e(i::Integer) 
   #    Js, Rs, Zs, z0 = get_neighbours(nlist, at, i) 
   #    e_i = eval_site(V, Rs, Zs, z0)
   #    release!(Js); release!(Rs); release!(Zs)
   #    return e_i 
   # end

   if isnothing(nlist)
      nlist = PairList(at, cutoff_radius(V))
   end

   E_tmp = Folds.map( domain, executor; init=ustrip(zero(V)) ) do i
      Js, Rs, Zs, z0 = get_neighbours(at, V, nlist, i) 
      e_i = eval_site(V, Rs, Zs, z0)
      release!(Js); release!(Rs); release!(Zs)
      e_i
   end
   E = zeros(eltype(E_tmp), size(at))
   E[domain] = E_tmp
   return E * energy_unit(V)
end


function AtomsCalculators.energy_forces!(f::AbstractVector, at, V::SitePotential; domain=1:length(at), nlist=nothing, kwargs...)
   # this is single threaded due to being non allocating
   @assert length(f) == length(at)
   if isnothing(nlist)  
      nlist = PairList(at, cutoff_radius(V))
   end
   E = ustrip(zero(V))
   for i in domain
      Js, Rs, Zs, z0 = get_neighbours(at, V, nlist, i) 
      v, dv = eval_grad_site(V, Rs, Zs, z0)
      E += v 
      f[Js] -= dv * force_unit(V)
      f[i]  += sum(dv) * force_unit(V)
      release!(Js); release!(Rs); release!(Zs); release!(dv)
   end
   return (; :energy => E * energy_unit(V), :force => f)
end

function AtomsCalculators.energy_forces(at, V::SitePotential; domain=1:length(at), executor=ThreadedEx(), ntasks=Threads.nthreads(), kwargs...)
   nlist = PairList(at, cutoff_radius(V))
   E_F = Folds.sum( collect(chunks(domain, ntasks)), executor; init=[zero(V), AtomsCalculators.zero_forces(at, V)] ) do (sub_domain, _)
      E = zero(V)
      f = AtomsCalculators.zero_forces(at, V)
      for i in sub_domain
         Js, Rs, Zs, z0 = get_neighbours(at, V, nlist, i) 
         v, dv = eval_grad_site(V, Rs, Zs, z0)
         E += v * energy_unit(V)
         f[Js] -= dv * force_unit(V)
         f[i]  += sum(dv) * force_unit(V)
         release!(Js); release!(Rs); release!(Zs); release!(dv)
      end
      [E, f]
   end
   return (; :energy => E_F[1], :force => E_F[2])
end

AtomsCalculators.forces(at, V::SitePotential; kwargs...) = AtomsCalculators.energy_forces(at, V; kwargs...)[:force]
AtomsCalculators.forces!(f, at, V::SitePotential; kwargs...) = AtomsCalculators.energy_forces!(f, at, V; kwargs...)[:force]
AtomsCalculators.calculate(::AtomsCalculators.Forces, at, V::SitePotential; kwargs...) = AtomsCalculators.forces(at, V; kwargs...)

function site_virial(V, dV, Rs) 
   if length(Rs) == 0
      return zero(SMatrix{3, 3, typeof(zero(V))})
   else
      return - sum( dv_i * 𝐫_i' for (dv_i, 𝐫_i) in zip(dV, Rs) ) * energy_unit(V)
   end
end


AtomsCalculators.@generate_interface function AtomsCalculators.virial(at, V::SitePotential; domain=1:length(at), executor=ThreadedEx(), kwargs...) 
   nlist = PairList(at, cutoff_radius(V))
   vir = Folds.sum( domain, executor; init=zero(SMatrix{3, 3, typeof(zero(V))}) ) do i 
      Js, Rs, Zs, z0 = get_neighbours(at, V, nlist, i) 
      _, dV = eval_grad_site(V, Rs, Zs, z0)
      vir_i = site_virial(V, dV, Rs)
      release!(Js); release!(Rs); release!(Zs); release!(dV)
      vir_i
   end
   return vir
end


function AtomsCalculators.energy_forces_virial(at, V::SitePotential; domain=1:length(at), executor=ThreadedEx(), ntasks=Threads.nthreads(), kwargs...)
   nlist = PairList(at, cutoff_radius(V))
   E_F_V = Folds.sum(
      collect(chunks(domain, ntasks)), 
      executor;
      init=[zero(V), AtomsCalculators.zero_forces(at, V), zero(SMatrix{3, 3, typeof(zero(V))})]
   ) do (sub_domain, _)
      E = zero(V)
      f = AtomsCalculators.zero_forces(at, V)
      vir = zero(SMatrix{3, 3, typeof(zero(V))})
      for i in sub_domain
         Js, Rs, Zs, z0 = get_neighbours(at, V, nlist, i) 
         v, dv = eval_grad_site(V, Rs, Zs, z0)
         E += v * energy_unit(V)
         f[Js] -= dv * force_unit(V)
         f[i]  += sum(dv) * force_unit(V)
         vir += site_virial(V, dv, Rs)
         release!(Js); release!(Rs); release!(Zs); release!(dv)
      end
      [E, f, vir]
   end
   return (; :energy => E_F_V[1], :force => E_F_V[2], :virial => E_F_V[3])
end