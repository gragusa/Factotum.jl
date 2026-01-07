using Documenter
using Factotum

makedocs(
    sitename = "Factotum.jl",
    modules = [Factotum],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://gragusa.github.io/Factotum.jl/stable/",
    ),
    pages = [
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "API Reference" => "api.md",
    ],
    checkdocs = :exports,
)

deploydocs(
    repo = "github.com/gragusa/Factotum.jl.git",
    devbranch = "master",
    push_preview = true,
)
