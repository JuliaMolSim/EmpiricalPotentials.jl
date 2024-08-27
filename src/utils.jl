
# this file has some experimental shared utilities that should 
# potentially be moved to AtomsCalculatorsUtilities


"""
   _z2i(zlist, z)

return an index in zlist or 0 if z was not found. If `nothing` should be 
returned instead of 0, then use `findfirst` instead. 
"""
function _z2i(zlist, z)
   for iz = 1:length(zlist)
       if zlist[iz] == z
           return iz
       end
   end
   return 0 
end 
