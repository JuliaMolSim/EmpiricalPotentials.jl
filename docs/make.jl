using EmpiricalPotentials
using Documenter

DocMeta.setdocmeta!(EmpiricalPotentials, :DocTestSetup, :(using EmpiricalPotentials); recursive=true)

makedocs(;
    modules=[EmpiricalPotentials],
    authors="Christoph Ortner <christohortner@gmail.com> and contributors",
    repo="https://github.com/JuliaMolSim/EmpiricalPotentials.jl/blob/{commit}{path}#{line}",
    sitename="EmpiricalPotentials.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JuliaMolSim.github.io/EmpiricalPotentials.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaMolSim/EmpiricalPotentials.jl",
    devbranch="main",
)
