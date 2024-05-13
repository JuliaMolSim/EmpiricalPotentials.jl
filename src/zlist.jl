


"""
`CatList` : simple data structure that stores a list of objects, the main 
use-case being converting between indices and categories. We use it to 
convert between chemical species (e.g. atomic numbers) and indices  in a list.

Constructor: 
```julia
list = CatList(list)
```
Indexing: 
```julia
list[i]       # returns the i-th category of the list
i2z(list, i)  # returns the i-th category of the list
z2i(list, z)  # returns the index of the category z in the list
```
"""
struct CatList{T}
   list::NTuple{N, T}

   function CatList(list1)
      list = ntuple(i -> list1[i], length(list1))
      return new{length(list), eltype(list)}(list)
   end
end

Base.length(zlist::CatList) = length(zlist.list)

Base.getindex(zlist::CatList, i::Integer) = zlist.list[i]

i2z(zlist::CatList, i::Integer) = Zs.list[i]

function z2i(zlist::CatList, z)
   for j = 1:length(zlist.list)
      if zlist.list[j] == z
         return j
      end
   end
   # not sure we want to raise an exception here. Better to just 
   # return an invalid index?  But -1 is not really invalid for some arrays
   error("z = $z not found in ZList $(Zs.list)")
   return -1 
end
