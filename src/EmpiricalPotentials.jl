module EmpiricalPotentials

using Unitful, ForwardDiff, StaticArrays, Bumper 

using ForwardDiff: Dual

import AtomsCalculators: energy_unit, length_unit, force_unit 

import AtomsCalculatorsUtilities
import AtomsCalculatorsUtilities.SitePotentials
import AtomsCalculatorsUtilities.SitePotentials: eval_site, eval_grad_site, 
                                 hessian_site, block_hessian_site,  
                                 cutoff_radius, SitePotential, 
                                 ad_block_hessian_site, ad_hessian_site, 
                                 cutoff_radius

import AtomsCalculatorsUtilities.PairPotentials: PairPotential, 
                                                 eval_pair 

include("utils.jl")                                                 

include("lennardjones.jl")
include("morse.jl")
include("zbl.jl")

include("stillingerweber.jl")



end
