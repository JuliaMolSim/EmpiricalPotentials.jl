module EmpiricalPotentials

using Unitful, ForwardDiff, StaticArrays, Bumper, StrideArrays

using ForwardDiff: Dual

import AtomsCalculators: energy_unit, length_unit, force_unit 

import AtomsCalculatorsUtilities
import AtomsCalculatorsUtilities.SitePotentials
import AtomsCalculatorsUtilities.SitePotentials: eval_site, eval_grad_site, 
                                 hessian_site, block_hessian_site,  
                                 cutoff_radius, SitePotential, 
                                 ad_block_hessian_site, ad_hessian_site


include("pairpotentials.jl")

include("stillingerweber.jl")



end
