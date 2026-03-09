# API Reference

This page documents all exported types and functions in Factotum.jl.

## Factor Model Types

```@docs
FactorModel
```

## Estimation Method Types

The estimation method is encoded as a type parameter on `FactorModel{E}`:

```@docs
AbstractEstimationMethod
PCA
EM
LeastSquares
```

### Querying the Estimation Method

```@docs
estimationmethod
```

### Example: Method-specific Dispatch

```@example api
using Factotum
using Random
Random.seed!(42)

X = randn(100, 20)

# Default (complete data) uses PCA
fm_pca = FactorModel(X, 3)
println("Method: ", estimationmethod(fm_pca))

# Explicit EM
fm_em = FactorModel(X, 3; method=:em)
println("Method: ", estimationmethod(fm_em))

# Explicit LS
fm_ls = FactorModel(X, 3; method=:ls)
println("Method: ", estimationmethod(fm_ls))

# Dispatch on estimation method type
describe_method(::FactorModel{PCA}) = "Fitted using Principal Component Analysis"
describe_method(::FactorModel{EM}) = "Fitted using EM Algorithm"
describe_method(::FactorModel{LeastSquares}) = "Fitted using Iterative Least Squares"

println(describe_method(fm_pca))
println(describe_method(fm_ls))
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
sdev
```

### Display

```@docs
describe
```

### R² Statistics

```@docs
TotalR2
ByFactorR2
total_r2
byfactor_r2
```

## Information Criteria

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

### Criterion Types

```@docs
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
```

### Criterion Functions

```@docs
informationcriteria
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
println("Estimation method: ", estimationmethod(fm))  # EM()
```

### Method Selection

The `method` parameter controls which estimation algorithm is used:

| Method | When to Use | Supports Missing Data | Supports Constraints |
|--------|-------------|----------------------|---------------------|
| `:pca` | Complete data, no constraints | No | No |
| `:em` | Missing data, no constraints | Yes | No |
| `:ls` | Constraints or missing data | Yes | Yes |
| `:auto` | Let Factotum choose (default) | - | - |

Auto-selection logic:
1. If `constraints` provided → `:ls`
2. If missing values (NaN) detected → `:em`
3. Otherwise → `:pca`

### EM/LS Algorithm Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `method` | `:auto` | Estimation method (`:pca`, `:em`, `:ls`, or `:auto`) |
| `init` | `nanmean` | Initial imputation function (`nanmean` or `nanmedian`) |
| `maxiter` | `1000` | Maximum iterations (EM or LS) |
| `tol` | `1e-8` | Convergence tolerance |
| `nt_min` | `10` | Minimum observations per series (LS only) |
| `orthonormalize` | `true` | For LS method, orthonormalize loadings via QR (set `false` for raw LS solution) |

## Constrained Estimation

The `:ls` method supports linear constraints on factor loadings for identification, sign normalization, or zero restrictions.

### Loading Constraints Types

```@docs
LoadingConstraints
normalize_loading
zero_loading
fix_loading
identity_loading
```

### Normalization Behavior

The default normalization is ``\Lambda'\Lambda = I`` (orthonormal loadings), which is enforced by PCA
and by the LS method when no constraints are provided (via QR decomposition after convergence).

When constraints are supplied, orthonormalization is **disabled** to preserve the constraint structure.
This means:

- **Sign normalization only** (e.g., `normalize_loading`): factors and loadings are the raw LS output.
  Loadings are *not* orthonormal and factors are *not* orthogonal.
- **Identity normalization** (`identity_loading`): imposes ``r^2`` constraints that fully resolve
  rotational indeterminacy. The loading submatrix for the named series equals ``I_r``.

### Creating Constraints

```@example api
using Factotum

# Fix loading of series 1 on factor 1 to 1.0 (sign normalization)
c1 = normalize_loading(1, 1; value=1.0)

# Set loading of series 5 on factor 2 to zero
c2 = zero_loading(5, 2)

# Fix the entire loading vector of series 4
c3 = fix_loading(4, [0.0, 0.0, 1.0])

# Identity normalization: series 1, 2, 3 define factors 1, 2, 3
c_id = identity_loading([1, 2, 3])

# Combine constraints
constraints = vcat(c1, c2)
```

### Using Constraints

```@example api
using Random
Random.seed!(42)
X = randn(100, 20)

# Fit constrained model
c = normalize_loading(1, 1; value=1.0)
fm = FactorModel(X, 3; constraints=c, scale=true)

# Verify constraint is satisfied
println("Loading[1,1] = ", round(loadings(fm)[1, 1], digits=6))
println("Method: ", estimationmethod(fm))
```

### Matrix Format for Constraints

For complex constraints, use a matrix where each row is `[series, R₁, ..., Rₖ, r]`:

```@example api
# Constraint: series 1, factor 1 loading = 1.0
# Format: [series, R₁, R₂, R₃, r] where R·λ = r
constraint_matrix = [
    1.0  1.0  0.0  0.0  1.0   # 1*λ₁ + 0*λ₂ + 0*λ₃ = 1.0
]
c = LoadingConstraints(constraint_matrix)
```

## Estimation Statistics

```@docs
EstimationStats
stats
tss
ssr
r2
nobs
```

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
