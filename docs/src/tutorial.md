# Guide

All examples below create their own data and are executed independently by
Documenter.

## Complete data: PCA and result extraction

Rows are observations and columns are variables. `demean=true` is the default;
use `scale=true` when variables have materially different units.

```@example guide-pca
using Factotum
using LinearAlgebra: eigvals
using Random

rng = MersenneTwister(11)
T, N = 100, 10
F0 = [sin.(range(0, 4pi; length=T)) cos.(range(0, 2pi; length=T))]
Lambda0 = randn(rng, N, 2)
Z = F0 * Lambda0' + 0.15 * randn(rng, T, N)

fm = FactorModel(Z, 2; scale=true)
(
    method = estimationmethod(fm),
    factors = size(factors(fm)),
    loadings = size(loadings(fm)),
    eigenvalues = eigvals(fm),
    standard_deviations = sdev(fm),
    explained_variance = explained_variance(fm),
    residual_norm = sqrt(ssr(fm)),
    observations = nobs(fm),
)
```

`loadings(fm)` is expressed on the working (possibly standardized) scale.
Recover coefficients in the original units with `original_units=true`.

```@example guide-units
using Factotum, Random

rng = MersenneTwister(12)
Z = randn(rng, 60, 5) .* [1 10 100 0.1 5]
fm = FactorModel(Z, 2; scale=true)
(standardized = loadings(fm), original = loadings(fm; original_units=true))
```

Use `describe` for a compact variance table and the scalar accessors for programmatic
work.

```@example guide-stats
using Factotum, Random

fm = FactorModel(randn(MersenneTwister(13), 50, 6), 2)
describe(fm)
(
    stats = stats(fm),
    tss = tss(fm),
    ssr = ssr(fm),
    series_r2 = r2(fm),
    residual_size = size(residuals(fm)),
)
```

## Choosing the number of factors

Fit once at the largest candidate number. Each criterion evaluates every rank
from zero through `kmax`.

```@example guide-selection
using Factotum, Random

rng = MersenneTwister(21)
F0 = randn(rng, 150, 2)
Lambda0 = randn(rng, 20, 2)
Z = F0 * Lambda0' + 0.3 * randn(rng, 150, 20)

kmax = 6
fm_max = FactorModel(Z, kmax; scale=true)
results = informationcriteria((IC1, IC2, IC3, PCp1, PCp2, PCp3,
                               AIC1, AIC2, AIC3, BIC1, BIC2, BIC3),
                              fm_max, kmax)
selected = [numfactors(ic) for ic in results]
first_curve = criterion(first(results))
(selected = selected, ranks = 0:kmax, IC1_values = first_curve,
 IC1_minimum = findmin(first(results)))
```

A criterion may also be called directly on a data matrix, for example
`IC2(Z, 6; scale=true)`.

## Missing data with EM

Encode missing entries as `NaN`. The default `method=:auto` detects them and
selects EM.

```@example guide-em
using Factotum, Random

rng = MersenneTwister(31)
F0 = randn(rng, 80, 2)
Z = F0 * randn(rng, 9, 2)' + 0.2 * randn(rng, 80, 9)
observed = rand(rng, size(Z)...) .> 0.12
Zmiss = copy(Z)
Zmiss[.!observed] .= NaN

fm = FactorModel(Zmiss, 2; scale=true, maxiter=1000, tol=1e-8)
(
    method = estimationmethod(fm),
    original_missing = count(isnan, Zmiss),
    nonmissing_used = nobs(fm),
    completed_working_matrix = count(isnan, fm.X̄),
)
```

The completed working matrix is stored internally. Its public field name is
`fm.X̄` (typed here directly because it is part of the model representation):

```@example guide-em-completed
using Factotum, Random

rng = MersenneTwister(32)
Z = randn(rng, 40, 5)
Z[1:4, 2] .= NaN
fm = FactorModel(Z, 2)
(missing_before = count(isnan, Z), missing_after = count(isnan, fm.X̄))
```

An information criterion can adapt the retained rank during EM; the supplied
factor count becomes `kmax`.

```@example guide-em-ic
using Factotum, Random

rng = MersenneTwister(33)
Z = randn(rng, 70, 8)
Z[1:7, 1] .= NaN
fm = FactorModel(Z, 4; ic=IC2, maxiter=500)
(selected_factors = numfactors(fm), maximum_considered = 4)
```

## Alternating least squares

LS can handle both complete and incomplete panels. With no restrictions it
orthonormalizes the final loadings by default.

```@example guide-ls
using Factotum, LinearAlgebra, Random

rng = MersenneTwister(41)
Z = randn(rng, 70, 8)
Z[1:8, 3] .= NaN
fm = FactorModel(Z, 2; method=:ls, nt_min=10)
(method = estimationmethod(fm), orthonormal = loadings(fm)'loadings(fm) ≈ I(2))
```

Set `orthonormalize=false` to keep the raw alternating-LS parameterization.

## Linear loading restrictions

Restrictions have the form ``R\lambda_i=q`` and automatically select LS. The
helper constructors cover the most common cases.

```@example guide-constraints
using Factotum, Random

rng = MersenneTwister(51)
Z = randn(rng, 100, 7)

c_equal = normalize_loading(1, 1; value=1.0) # lambda[1,1] = 1
c_zero  = zero_loading(2, 2)                 # lambda[2,2] = 0
c_row   = fix_loading(3, [0.0, 1.0])         # entire third row
constraints = vcat(c_equal, c_zero, c_row)

fm = FactorModel(Z, 2; constraints=constraints)
(method = estimationmethod(fm), constrained_rows = loadings(fm)[1:3, :])
```

`identity_loading` names the factors by fixing an ``r\times r`` loading block to
the identity.

```@example guide-identity
using Factotum, LinearAlgebra, Random

rng = MersenneTwister(52)
Z = randn(rng, 100, 7)
fm = FactorModel(Z, 2; constraints=identity_loading([1, 2]))
loadings(fm)[1:2, :] ≈ I(2)
```

For general restrictions, each row of a matrix is
`[series, R1, ..., Rr, q]`.

```@example guide-constraint-matrix
using Factotum, Random

# Series 1: lambda_11 + lambda_12 = 0.5
# Series 4: lambda_42 = 0
C = [1.0  1.0  1.0  0.5;
     4.0  0.0  1.0  0.0]
constraints = LoadingConstraints(C)
fm = FactorModel(randn(MersenneTwister(53), 80, 6), 2;
                 constraints=constraints)
(sum_first = sum(loadings(fm)[1, :]), fourth_second = loadings(fm)[4, 2])
```

Restrictions and missing observations can be combined in the same LS fit.

## Factor views

A view reuses an existing fit without recomputing it. An integer selects the
first factors; a unit range selects any consecutive block.

```@example guide-views
using Factotum, Random

fm = FactorModel(randn(MersenneTwister(61), 90, 10), 5)
first_two = view(fm, 2)
middle = view(fm, 2:4)
(first = (size(factors(first_two)), size(loadings(first_two))),
 middle = (numfactors(middle), size(loadings(middle))))
```

## Total and factor-specific R² tables

`total_r2` ranks variables by their fit in a model. `byfactor_r2` attributes fit
to individual orthogonal factors. Both results implement the Tables.jl
interface, so they can be passed to downstream table packages.

```@example guide-r2
using Factotum, Random

rng = MersenneTwister(71)
F0 = randn(rng, 100, 2)
Z = F0 * randn(rng, 6, 2)' + 0.4 * randn(rng, 100, 6)
fm = FactorModel(Z, 2)
names = ["series_$i" for i in 1:6]

total = total_r2(fm; varnames=names, show_all=true)
by_factor = byfactor_r2(fm; varnames=names,
                        show_all=true)
(total_columns = Factotum.Tables.columnnames(total),
 byfactor_columns = Factotum.Tables.columnnames(by_factor))
```

## Method selection and dispatch

The estimation algorithm is a type parameter, enabling ordinary Julia dispatch.

```@example guide-dispatch
using Factotum, Random

label(::FactorModel{PCA}) = "complete-data PCA"
label(::FactorModel{EM}) = "missing-data EM"
label(::FactorModel{LeastSquares}) = "alternating least squares"

rng = MersenneTwister(81)
complete = randn(rng, 40, 5)
incomplete = copy(complete); incomplete[1, 1] = NaN
constrained = normalize_loading(1, 1)

(label(FactorModel(complete, 2)),
 label(FactorModel(incomplete, 2)),
 label(FactorModel(complete, 2; constraints=constrained)))
```
