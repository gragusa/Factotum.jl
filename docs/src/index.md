# Factotum.jl

`Factotum.jl` estimates **static approximate factor models** from a matrix whose
rows are observations and whose columns are variables. It covers the complete
workflow: preprocessing, estimation by PCA, estimation with missing observations,
restricted loadings, diagnostics, variance attribution, and selection of the
number of factors.

## What the package can do

- estimate factors and loadings by principal components;
- choose EM automatically when observations are encoded as `NaN`;
- estimate by alternating least squares, with or without missing data;
- impose linear restrictions, zero restrictions, and named-factor
  normalizations on the loadings;
- select the number of factors with IC, PCp, AIC, and BIC criteria;
- report residuals, total and series-level ``R^2``, explained variance, and
  factor-specific ``R^2``;
- return lightweight views of a fit containing only selected factors.

## Installation

```julia
using Pkg
Pkg.add("Factotum")
```

## A first reproducible fit

The example constructs a two-factor panel, fits it, and checks the dimensions of
the recovered low-rank representation. Every `@example` in this manual is run by
Documenter when the site is built.

```@example home-fit
using Factotum
using Random

rng = MersenneTwister(2026)
T, N, r = 120, 12, 2
F0 = randn(rng, T, r)
Lambda0 = randn(rng, N, r)
Z = F0 * Lambda0' + 0.25 * randn(rng, T, N)

fm = FactorModel(Z, r; scale=true)
(method = estimationmethod(fm),
 factors = size(factors(fm)),
 loadings = size(loadings(fm)),
 explained = sum(explained_variance(fm)))
```

The fitted common component is `factors(fm) * loadings(fm)'`; `residuals(fm)`
is what remains after subtracting it from the centered and, when requested,
scaled panel.

## Where to go next

- [Mathematical introduction](@ref) explains the model, PCA objective,
  normalization, missing-data algorithms, restrictions, and factor-number
  criteria.
- [Guide](@ref) gives self-contained examples of every major workflow.
- [API reference](@ref) lists all public types and functions.

```@contents
Pages = ["mathematics.md", "tutorial.md", "api.md"]
Depth = 2
```

## References

- Bai, J. and Ng, S. (2002). Determining the number of factors in approximate
  factor models. *Econometrica*, 70(1), 191–221.
