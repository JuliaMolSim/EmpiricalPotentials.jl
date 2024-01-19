var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = EmpiricalPotentials","category":"page"},{"location":"#EmpiricalPotentials","page":"Home","title":"EmpiricalPotentials","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for EmpiricalPotentials.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [EmpiricalPotentials]","category":"page"},{"location":"#EmpiricalPotentials.PairPotential","page":"Home","title":"EmpiricalPotentials.PairPotential","text":"PairPotential:abstractsupertype for pair potentials\n\n\n\n\n\n","category":"type"},{"location":"#EmpiricalPotentials.SitePotential","page":"Home","title":"EmpiricalPotentials.SitePotential","text":"SitePotential:abstractsupertype for generic site potentials. Concrete subtypes  should overload the cutoff_radius, eval_site and eval_grad_site methods. \n\n\n\n\n\n","category":"type"},{"location":"#EmpiricalPotentials.StillingerWeber","page":"Home","title":"EmpiricalPotentials.StillingerWeber","text":"Stillinger-Weber potential with parameters for Si.\n\nFunctional form and default parameters match the original SW potential from [Stillinger/Weber, PRB 1985].\n\nThe StillingerWeber type can also by \"abused\" to generate arbitrary bond-angle potentials of the form    Σᵢⱼ V₂(rᵢⱼ) + Σᵢⱼₖ V₃(rᵢⱼ) V₃(rᵢₖ) (cos Θᵢⱼₖ + 1/3)²\n\nConstructor admits the following key-word parameters: ϵ=2.1675, σ = 2.0951, A=7.049556277, B=0.6022245584,    p = 4, a = 1.8, λ=21.0, γ=1.20, atom_number=14\n\nwhich enter the potential as follows:\n\nV2(r) = 0.5 * ϵ * A * (B * (r/σ)^(-p) - 1.0) * exp(1.0 / (r/σ - a))\nV3(r) = sqrt(ϵ * λ) * exp(γ / (r/σ - a))\n\nThe brittle keyword can be used to switch to the parameters λ to 42.0,  which is appropriate to simulate brittle fracture. (need reference for this)\n\n\n\n\n\n","category":"type"},{"location":"#EmpiricalPotentials.cutoff_radius","page":"Home","title":"EmpiricalPotentials.cutoff_radius","text":"write docs...\n\n\n\n\n\n","category":"function"},{"location":"#EmpiricalPotentials.eval_grad_site","page":"Home","title":"EmpiricalPotentials.eval_grad_site","text":"If V <: SitePotential then it should implement the method\n\ndv = eval_grad_site(V, Rs, Zs, z0)\n\nwhere Rs::AbstractVector{<: SVector{3}} and Zs::AbstractVector of  atom ids (e.g., atomic numbers), while z0 is a single atom id. \n\nThe output dv should be an AbstractVector containing   SVector{3,T} blocks.\n\n\n\n\n\n","category":"function"},{"location":"#EmpiricalPotentials.eval_site","page":"Home","title":"EmpiricalPotentials.eval_site","text":"If V <: SitePotential then it should implement the method\n\nval = eval_site(V, Rs, Zs, z0)\n\nwhere Rs::AbstractVector{<: SVector{3}} and Zs::AbstractVector of atom ids  (e.g. atomic numbers), while z0 is a single atom id. \n\nThe output val should be a single number, namely the site energy.\n\n\n\n\n\n","category":"function"},{"location":"#EmpiricalPotentials.get_neighbours-NTuple{4, Any}","page":"Home","title":"EmpiricalPotentials.get_neighbours","text":"get_neighbours(nlist::PairList, at, i::Integer) -> Js, Rs, Zs, z0\n\n\n\n\n\n","category":"method"},{"location":"#EmpiricalPotentials.precon","page":"Home","title":"EmpiricalPotentials.precon","text":"If V <: SitePotential then it can implement the method\n\nPblock = precon(V, Rs, Zs, z0)\n\nwhere Rs::AbstractVector{<: SVector{3}} and Zs::AbstractVector of  atom ids (e.g., atomic numbers), while z0 is a single atom id.  The output Pblock should be an AbstractMatrix containing  SMatrix{3,3,T} blocks. \n\nUnlike eval_site and eval_grad_site, this method is optional. It  can be used to speedup geometry optimization, sampling and related tasks. \n\n\n\n\n\n","category":"function"},{"location":"#EmpiricalPotentials.sw_bondangle-Tuple{Any, Any}","page":"Home","title":"EmpiricalPotentials.sw_bondangle","text":"sw_bondangle(S1, S2) -> (dot(S1, S2) + 1.0/3.0)^2\n\nnot this assumes that S1, S2 are normalised\nsee sw_bondangle_d for the derivative\n\n\n\n\n\n","category":"method"},{"location":"#EmpiricalPotentials.sw_bondangle_d-NTuple{4, Any}","page":"Home","title":"EmpiricalPotentials.sw_bondangle_d","text":"b := sw_bondangle(S1, S2) then\n\nsw_bondangle_d(S1, S2, r1, r2) -> b, db1, db2\n\nwhere dbi is the derivative of b w.r.t. Ri where Si= Ri/ri.\n\n\n\n\n\n","category":"method"}]
}
