
# this needs to be replaced with a suitable interface to ASE, possibly 
# via an extension 
#
# # Use Requires.jl to provide the ASE EAM constructor.
# function __init__()
#    @require ASE="51974c44-a7ed-5088-b8be-3e78c8ba416c" @eval eam_from_ase(
#          filename::AbstractString; kwargs...) =
#          (
#             eam = ASE.Models.EAM(filename).po; # Use ASE to create calculator
#             EAM(eam.nr, eam.dr, eam.nrho, eam.drho, eam.cutoff, eam.Z, eam.density_data,
#                 eam.embedded_data, eam.rphi_data;
#                 kwargs...)
#          )
# end
