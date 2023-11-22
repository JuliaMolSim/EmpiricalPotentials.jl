
using ObjectPools: acquire!, release! 


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



"""
`get_neighbours(nlist::PairList, at::Atoms, i::Integer) -> Js, Rs, Zs, z0`
"""
function get_neighbours(nlist, at, i)
   Js, Rs = NeighbourLists.neigs(nlist, i)
   Zs = at.Z[Js] 
   z0 = at.Z[i] 
   return Js, Rs, Zs, z0 
end


function potential_energy(V::SitePotential, at, nlist; domain = 1:length(at)) 
   # this has a problem: if domain is empty, then `sum` needs an initializer 
   # and I don't know how to infer the type of the output. I JuLIP this was 
   # solved via `fltype(V)` or something like that ... 

   function _site_e(i::Integer) 
      Js, Rs, Zs, z0 = get_neighbours(nlist, at, i) 
      e_i = eval_site(V, Rs, Zs, z0)
      release!(Js); release!(Rs); release!(Zs)
      return e_i 
   end

   return sum(_site_e, domain)
end


function energy_and_forces!(f, V::SitePotential, at, nlist; domain = 1:length(at))   

   # same problem - unclear how to initialize energy 
   # I'm guess it should be the same as for the forces, but is this 
   # really clean? 
   E = zero(eltype(eltype(f)))

   for i in domain
      Js, Rs, Zs, z0 = get_neighbours(nlist, at, i) 
      v, dv = eval_grad_site(V, Rs, Zs, z0)
      E += v 
      for k in eachindex(Js)
          f[Js[k]] -= tmp.dv[k]
          f[i]     += tmp.dv[k]
      end
      release!(Js); release!(Rs); release!(Zs); release!(dV)
   end

   return E, f
end



function site_virial(dV, Rs) 
   TV = eltype(eltype(dV))
   TR = eltype(eltype(Rs))
   T = promotetype(TV, TR)
   if length(Rs) == 0 
      return zero(SMatrix{3, 3, T})
   else
      return - sum( dv_i * 𝐫_i' for (dv_i, 𝐫_i) in zip(dV, Rs) )
   end
end


function virial(V::SitePotential, at, nlist; domain = 1:length(at)) 

   # same problem with initialization of sum again 

   function _site_virial(i::Integer) 
      Js, Rs, Zs, z0 = get_neighbours(nlist, at, i) 
      _, dV = eval_grad_site(V, Rs, Zs, z0)
      vir_i = site_virial(dV, Rs)
      release!(Js); release!(Rs); release!(Zs); release!(dV)
      return vir_i 
   end

   return sum(_site_virial, domain)
end
