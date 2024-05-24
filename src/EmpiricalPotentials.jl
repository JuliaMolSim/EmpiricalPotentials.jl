module EmpiricalPotentials

using Unitful, ForwardDiff, StaticArrays, Bumper, StrideArrays

using ForwardDiff: Dual

import AtomsCalculatorsUtilities
import AtomsCalculatorsUtilities.SitePotentials
import AtomsCalculatorsUtilities.SitePotentials: eval_site, eval_grad_site, 
                                 hessian_site, block_hessian_site,  
                                 cutoff_radius, energy_unit, length_unit

using AtomsCalculatorsUtilities.SitePotentials: SitePotential, 
                        ad_block_hessian_site, ad_hessian_site


include("pairpotentials.jl")

include("stillingerweber.jl")



end
