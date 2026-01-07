# API Reference

This page documents all exported types and functions in Factotum.jl.

## Factor Model Types

```@docs
FactorModel
```

## Factor Model Functions

### Estimation

```@docs
FactorModel(::AbstractMatrix, ::Any)
```

### Extraction

```@docs
factors
loadings
numfactors
explained_variance
```

### Display

```@docs
describe
```

## Information Criteria

### Criterion Types

Factotum.jl provides 12 information criteria for model selection:

| Type | Description |
|------|-------------|
| `IC1`, `IC2`, `IC3` | Bai & Ng (2002) information criteria |
| `PCp1`, `PCp2`, `PCp3` | Panel Cp criteria |
| `AIC1`, `AIC2`, `AIC3` | AIC-type criteria |
| `BIC1`, `BIC2`, `BIC3` | BIC-type criteria |

Each criterion can be called as a function:

```@example api
using Factotum
using Random
Random.seed!(42)

X = randn(100, 20)
fm = FactorModel(X, 8; scale=true)

# Compute IC1 criterion
ic1 = IC1(fm, 8)
println("Optimal number of factors: ", numfactors(ic1))
```

### Computing Multiple Criteria

```@example api
# Compute multiple criteria simultaneously
criteria = informationcriteria((IC1, IC2, IC3, BIC1), fm, 8)

# Find optimal r for each
for result in findmin(criteria)
    println(result)
end
```

### Criterion Functions

```@docs
criterion
```

### Working with Criterion Results

```@example api
# Get the criterion values
values = criterion(ic1)
println("Values for r=0 to 8: ", round.(values, digits=3))

# Find minimum
result = findmin(ic1)
println("Minimum at r=$(result.r) with value=$(round(result[1], digits=3))")
```

## Views

Create views into a factor model with a subset of factors:

```@example api
fm = FactorModel(randn(100, 20), 5)

# View with first 3 factors
fm3 = view(fm, 3)
println("Factors in view: ", numfactors(fm3))

# View with factors 2-4
fm24 = view(fm, 2:4)
println("Loadings shape: ", size(loadings(fm24)))
```

## Missing Data Support

The package automatically handles missing values (NaN) using an EM algorithm:

```@example api
using Random
Random.seed!(42)

# Create data with missing values
X = randn(100, 20)
X[rand(100, 20) .< 0.1] .= NaN

# Fit model - EM algorithm is used automatically
fm = FactorModel(X, 3; scale=true)
println("Model fitted with $(numfactors(fm)) factors")
```

### EM Algorithm Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `em` | `false` | Force EM algorithm (auto-enabled if NaN detected) |
| `init` | `nanmean` | Initial imputation function (`nanmean` or `nanmedian`) |
| `maxiter` | `1000` | Maximum EM iterations |
| `tol` | `1e-8` | Convergence tolerance |

## Complete API Index

### Exported Functions

```@index
Modules = [Factotum]
Order = [:function]
```

### Exported Types

```@index
Modules = [Factotum]
Order = [:type]
```
