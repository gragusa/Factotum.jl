using Documenter
using Factotum

DocMeta.setdocmeta!(Factotum, :DocTestSetup, :(using Factotum, LinearAlgebra); recursive = true)

makedocs(;
    source = joinpath(@__DIR__, "src"),
    sitename = "Factotum.jl",
    modules = [Factotum],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        collapselevel = 3,
        canonical = "https://gragusa.github.io/Factotum.jl/stable/"
    ),
    pages = [
        "Home" => "index.md",
        "Mathematical introduction" => "mathematics.md",
        "Guide" => "tutorial.md",
        "API Reference" => "api.md"
    ],
    checkdocs = :exports
)

deploydocs(
    repo = "github.com/gragusa/Factotum.jl.git",
    devbranch = "master",
    push_preview = true
)
