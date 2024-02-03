

abstract type AbstractZList end

"""
`ZList` and `SZList{NZ}` : simple data structures that store a list
of species and convert between atomic numbers and the index in the list.
Can be constructed via
* `ZList(zors)` : where `zors` is an Integer  or `Symbol` (single species)
* `ZList(zs1, zs2, ..., zsn)`
* `ZList([sz1, zs2, ..., zsn])`
* All of these take a kwarg `static = {true, false}`; if `true`, then `ZList`
will return a `SZList{NZ}` for (possibly) faster access.
"""
struct ZList <: AbstractZList
   list::Vector{AtomicNumber}
end

Base.length(zlist::AbstractZList) = length(zlist.list)


struct SZList{N} <: AbstractZList
   list::SVector{N, AtomicNumber}
end

function ZList(zlist::AbstractVector{<: Number};
               static = false, sorted=true)
   sortfun = sorted ? sort : identity
   return (static ? SZList(SVector( (AtomicNumber.(sortfun(zlist)))... ))
                  :  ZList( convert(Vector{AtomicNumber}, sortfun(zlist)) ))
end

ZList(s::Symbol; kwargs...) =
      ZList( [ atomic_number(s) ]; kwargs... )

ZList(S::AbstractVector{Symbol}; kwargs...) =
      ZList( atomic_number.(S); kwargs... )

ZList(args...; kwargs... ) =
      ZList( [args...]; kwargs...)


i2z(Zs::AbstractZList, i::Integer) = Zs.list[i]

function z2i(Zs::AbstractZList, z::AtomicNumber)
   if Zs.list[1] == JuLIP.Chemistry.__zAny__
      return 1
   end
   for j = 1:length(Zs.list)
      if Zs.list[j] == z
         return j
      end
   end
   error("z = $z not found in ZList $(Zs.list)")
end

zlist(V) = V.zlist
i2z(V, i::Integer) = i2z(zlist(V), i)
z2i(V, z::AtomicNumber ) = z2i(zlist(V), z)
numz(V) = length(zlist(V))

write_dict(zlist::ZList) = Dict("__id__" => "JuLIP_ZList",
                                  "list" => Int.(zlist.list))

read_dict(::Val{:JuLIP_ZList}, D::Dict) = ZList(D)
ZList(D::Dict) = ZList(D["list"])

write_dict(zlist::SZList) = Dict("__id__" => "JuLIP_SZList",
                                 "list" => Int.(zlist.list))
read_dict(::Val{:JuLIP_SZList}, D::Dict) = SZList(D)
SZList(D::Dict) = ZList([D["list"]...], static = true)