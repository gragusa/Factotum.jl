# Factotum.jl

[![Build Status](https://github.com/gragusa/Factotum.jl/workflows/CI/badge.svg)](https://github.com/gragusa/Factotum.jl/actions)

A Julia package for estimating **static factor models** using Principal Component Analysis (PCA). Factotum.jl provides tools for dimensionality reduction, factor extraction, and model selection via multiple information criteria.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/gragusa/Factotum.jl")
```

## Quick Start

```julia
using Factotum

# Generate some data
X = randn(200, 50)  # 200 observations, 50 variables

# Fit a factor model with 5 factors
fm = FactorModel(X, 5; demean=true, scale=true)

# View model summary
describe(fm)
```

Output:
```
Static Factor Model
Dimensions of X..........: (200, 50)
Number of factors........: 5
Factors calculated by....: Principal Component
Factors' importance:
...
```

## The Factor Model

Factotum.jl estimates the static factor model:

```
X = F * Λ' + ε
```

where:
- **X** is a T × n data matrix (T observations, n variables)
- **F** is a T × r matrix of factors
- **Λ** is an n × r matrix of factor loadings (with Λ'Λ/n = I)
- **ε** is a T × n matrix of idiosyncratic errors
- **r** is the number of factors

## Basic Usage

### Fitting a Factor Model

```julia
using Factotum

# Load or generate your data
X = randn(100, 20)

# Fit with specified number of factors
fm = FactorModel(X, 5)

# Options:
# - demean: subtract column means (default: true)
# - scale: standardize columns to unit variance (default: false)
fm = FactorModel(X, 5; demean=true, scale=true)
```

### Accessing Results

```julia
# Get the estimated factors (T × r matrix)
F = factors(fm)

# Get the factor loadings (n × r matrix)
Λ = loadings(fm)

# Number of factors
r = numfactors(fm)

# Proportion of variance explained by each factor
ev = explained_variance(fm)

# Model summary
describe(fm)
```

### Working with Factor Subsets

You can create views of a model with fewer factors without refitting:

```julia
fm = FactorModel(X, 10)

# View with first 3 factors only
fm3 = view(fm, 3)

# View with factors 2 through 5
fm_subset = view(fm, 2:5)
```

## Model Selection with Information Criteria

Factotum.jl implements the information criteria from Bai & Ng (2002) for selecting the optimal number of factors.

### Available Criteria

| Criterion | Description |
|-----------|-------------|
| `IC1`, `IC2`, `IC3` | Information criteria (Bai & Ng, 2002) |
| `PCp1`, `PCp2`, `PCp3` | Panel criteria variants |
| `AIC1`, `AIC2`, `AIC3` | Akaike information criterion variants |
| `BIC1`, `BIC2`, `BIC3` | Bayesian information criterion variants |

### Using Information Criteria

```julia
using Factotum

# Simulate a factor model with 3 true factors
function simulate_factormodel(r, T, N)
    F = randn(T, r)
    Λ = randn(r, N)
    e = sqrt(r) .* randn(T, N)
    F * Λ .+ e
end

X = simulate_factormodel(3, 200, 50)

# Fit model with maximum number of factors to consider
fm = FactorModel(X, 10; scale=true)

# Compute a single criterion
ic1 = IC1(fm, 10)

# Display the criterion values (highlights minimum)
ic1
```

Output:
```
┌──────────────┬───────────┐
│ # of factors │ Criterion │
├──────────────┼───────────┤
│            0 │     0.693 │
│            1 │     0.512 │
│            2 │     0.298 │
│            3 │    0.0891 │  ← minimum (highlighted)
│            4 │     0.102 │
│            5 │     0.115 │
│          ... │       ... │
└──────────────┴───────────┘
```

### Comparing Multiple Criteria

```julia
# Compute multiple criteria at once
ics = Factotum.informationcriteria((IC1, IC2, BIC1), fm, 10)

# Display comparison table
ics

# Get optimal number of factors for each criterion
findmin(ics)

# Get optimal r from a single criterion
optimal_r = numfactors(ic1)
```

### Direct Matrix Interface

You can also compute criteria directly from a data matrix:

```julia
# No need to create FactorModel first
ic = IC1(X, 10; scale=true)
```

## API Reference

### Types

- `FactorModel` - Main type holding the estimated factor model
- `FactorModelView` - Lightweight view into a subset of factors

### Constructors

```julia
FactorModel(X::Matrix, r::Int; demean=true, scale=false)
```

### Accessor Functions

| Function | Description |
|----------|-------------|
| `factors(fm)` | Extract the T × r factor matrix |
| `loadings(fm)` | Extract the n × r loadings matrix |
| `numfactors(fm)` | Number of factors |
| `explained_variance(fm)` | Proportion of variance explained |
| `describe(fm)` | Print model summary |

### Information Criteria

```julia
# Single criterion
IC1(fm, kmax)    # or IC2, IC3, PCp1, ..., BIC3

# Multiple criteria
Factotum.informationcriteria((IC1, IC2, BIC1), fm, kmax)

# Find optimal number of factors
numfactors(ic)   # returns optimal r
findmin(ic)      # returns (minimum_value, optimal_r)
```

## Example: Factor Model for Panel Data

```julia
using Factotum, Random

Random.seed!(123)

# Simulate a dataset with 3 underlying factors
T, N, r_true = 250, 100, 3

F_true = randn(T, r_true)
Λ_true = randn(r_true, N)
ε = 0.5 * randn(T, N)
X = F_true * Λ_true + ε

# Estimate factor model (testing up to 10 factors)
fm = FactorModel(X, 10; scale=true)

# Use information criteria to select number of factors
ic = IC1(fm, 10)
r_hat = numfactors(ic)

println("True number of factors: $r_true")
println("Estimated number of factors: $r_hat")

# Extract the estimated factors
F_hat = factors(view(fm, r_hat))
```

## References

- Bai, J., & Ng, S. (2002). Determining the number of factors in approximate factor models. *Econometrica*, 70(1), 191-221.

## License

MIT License - see LICENSE.md for details.
