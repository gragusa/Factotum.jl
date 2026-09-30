# API reference

The guide demonstrates the public interface in context. This page collects the
corresponding docstrings.

## Model and estimation methods

```@docs
FactorModel
AbstractEstimationMethod
PCA
EM
LeastSquares
estimationmethod
```

## Extracting a fit

```@docs
factors
loadings
canonical_correlation
numfactors
sdev
total_variance
explained_variance
residuals
describe
```

`LinearAlgebra.eigvals`, `Base.size`, `Base.view`, and `Base.show` are also
implemented for factor models.

## Estimation statistics and R²

```@docs
EstimationStats
stats
tss
ssr
r2
nobs
TotalR2
ByFactorR2
total_r2
byfactor_r2
```

## Factor-number criteria

All criterion objects support `numfactors`, `criterion`, and `findmin`.

```@docs
Factotum.AbstractInformationCriterion
IC1
IC2
IC3
PCp1
PCp2
PCp3
AIC1
AIC2
AIC3
BIC1
BIC2
BIC3
informationcriteria
criterion
```

## Loading restrictions

```@docs
LoadingConstraints
normalize_loading
zero_loading
fix_loading
identity_loading
```

## Index

```@index
Modules = [Factotum]
```
