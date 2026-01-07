# Tutorial

This tutorial walks through the main features of Factotum.jl, from basic factor model estimation to advanced model selection and missing data handling.

## Background: Factor Models

Factor models represent high-dimensional data as a combination of a few latent factors plus idiosyncratic noise:

```math
X_{it} = \lambda_i' F_t + \varepsilon_{it}
```

where:
- ``X`` is a ``T \times n`` data matrix (``T`` time periods, ``n`` variables)
- ``F`` is a ``T \times r`` matrix of latent factors
- ``\Lambda`` is an ``n \times r`` matrix of factor loadings
- ``\varepsilon`` is the idiosyncratic error term

Factotum.jl estimates ``F`` and ``\Lambda`` using Principal Component Analysis (PCA).

## Basic Usage

### Loading Macroeconomic Data

We'll use a panel of 233 quarterly US macroeconomic time series spanning 1959-2024. The data includes output, employment, prices, interest rates, and money supply indicators, transformed for stationarity and locally demeaned.

```@example tutorial
using Factotum
using CSV, DataFrames
using LinearAlgebra
using Statistics

# Load quarterly US macroeconomic data
datapath = joinpath(dirname(dirname(pathof(Factotum))), "test", "data", "macrodata.csv")
df = CSV.read(datapath, DataFrame)
dates = df.DATE
X_full = Matrix{Float64}(df[:, 2:end])
varnames = names(df)[2:end]

println("Full dataset: $(size(X_full, 1)) quarters × $(size(X_full, 2)) variables")
println("Date range: $(dates[1]) to $(dates[end])")
```

### Creating a Balanced Panel

For examples that don't use the EM algorithm, we create a balanced panel by dropping series with any missing values:

```@example tutorial
# Drop series with any missing values
complete_cols = [all(x -> !isnan(x), col) for col in eachcol(X_full)]
X = X_full[:, complete_cols]
varnames_complete = varnames[complete_cols]

println("Balanced panel: $(size(X, 1)) quarters × $(size(X, 2)) variables")
```

### Fitting a Factor Model

Use `FactorModel` to estimate factors from the data:

```@example tutorial
# Fit a factor model with 5 factors
fm = FactorModel(X, 5)
Factotum.describe(fm)
```

### Preprocessing Options

Control how data is centered and scaled:

```@example tutorial
# Center only (default)
fm_centered = FactorModel(X, 3; demean=true, scale=false)

# Center and standardize to unit variance
fm_scaled = FactorModel(X, 3; demean=true, scale=true)

# Use Bessel's correction for standard deviation
fm_corrected = FactorModel(X, 3; scale=true, corrected=true)
nothing # hide
```

### Extracting Results

Access the estimated components:

```@example tutorial
# Extract factors (T × r matrix)
F = factors(fm)
println("Factors size: ", size(F))

# Extract loadings (n × r matrix)
Lambda = loadings(fm)
println("Loadings size: ", size(Lambda))

# Number of factors
println("Number of factors: ", numfactors(fm))

# Explained variance by each factor
ev = explained_variance(fm)
println("Explained variance: ", round.(ev, digits=3))

# Cumulative explained variance
println("Cumulative variance: ", round.(cumsum(ev), digits=3))

# First factor explains the most variance (a common "level" factor in macro data)
println("First 5 factors explain ", round(100 * sum(ev), digits=1), "% of total variance")
```

### Eigenvalues and Standard Deviations

```@example tutorial
using LinearAlgebra: eigvals

# Eigenvalues
lambda = eigvals(fm)
println("Eigenvalues: ", round.(lambda, digits=2))

# Standard deviations of factors
sd = Factotum.sdev(fm)
println("Standard deviations: ", round.(sd, digits=4))
```

## Model Selection with Information Criteria

A critical question in factor analysis is: *how many factors should we use?* Factotum.jl provides several information criteria to help answer this question.

### Available Criteria

The package implements 12 information criteria from the literature:

| Criterion | Description |
|-----------|-------------|
| `IC1`, `IC2`, `IC3` | Bai & Ng (2002) information criteria |
| `PCp1`, `PCp2`, `PCp3` | Panel Cp criteria |
| `AIC1`, `AIC2`, `AIC3` | AIC-type criteria |
| `BIC1`, `BIC2`, `BIC3` | BIC-type criteria |

### Using Information Criteria

First, fit a model with the maximum number of factors you want to consider:

```@example tutorial
# Fit model with up to 10 factors
kmax = 10
fm_full = FactorModel(X, kmax; scale=true)
```

Compute an information criterion:

```@example tutorial
# Compute IC1 for 0 to 10 factors
ic1 = IC1(fm_full, kmax)
```

Find the optimal number of factors:

```@example tutorial
# Get optimal r (minimizes criterion)
r_optimal = numfactors(ic1)
println("Optimal number of factors (IC1): ", r_optimal)

# Get detailed results
result = findmin(ic1)
println("Minimum IC1 value: ", round(result[1], digits=4), " at r = ", result.r)
```

Access the criterion values:

```@example tutorial
# Get all criterion values
values = criterion(ic1)
println("IC1 values for r = 0 to $kmax:")
for (r, v) in enumerate(values)
    println("  r = $(r-1): ", round(v, digits=4))
end
```

### Comparing Multiple Criteria

Compare several criteria simultaneously:

```@example tutorial
# Compute multiple criteria at once
criteria = informationcriteria((IC1, IC2, IC3, BIC1, BIC2, BIC3), fm_full, kmax)

# Display comparison table
criteria
```

Find optimal number of factors for each criterion:

```@example tutorial
results = findmin(criteria)
for r in results
    println(r)
end
```

### Direct Computation from Data

You can compute information criteria directly from a data matrix:

```@example tutorial
# Compute IC1 directly from data
ic1_direct = IC1(X, kmax; scale=true)
println("Optimal r (direct): ", numfactors(ic1_direct))
```

## Working with Subsets of Factors

Use `view` to create lightweight views into a factor model with fewer factors:

```@example tutorial
# Full model with 5 factors
fm5 = FactorModel(X, 5; scale=true)

# View with only first 3 factors (no recomputation)
fm3_view = view(fm5, 3)
println("Number of factors in view: ", numfactors(fm3_view))

# View with factors 2 to 4
fm24_view = view(fm5, 2:4)
println("Factors 2-4 loadings size: ", size(loadings(fm24_view)))
```

## Handling Missing Data

Factotum.jl supports data with missing values (represented as `NaN`) using an EM algorithm.

### Automatic Detection

The full macroeconomic dataset contains missing values due to different series start dates and data transformations (e.g., differencing). Missing values are automatically detected and handled:

```@example tutorial
# Use the full dataset with missing values
n_missing = sum(isnan.(X_full))
pct_missing = round(100 * n_missing / length(X_full), digits=1)

println("Full dataset: $(size(X_full, 1)) quarters × $(size(X_full, 2)) variables")
println("Missing values: $n_missing ($pct_missing%)")

# Fit model - EM algorithm is used automatically
fm_em = FactorModel(X_full, 5; scale=true)
Factotum.describe(fm_em)
```

### EM Algorithm Options

Control the EM algorithm behavior:

```@example tutorial
# Custom EM settings
fm_em_custom = FactorModel(X_full, 5;
    scale=true,
    em=true,              # Force EM (auto-enabled with NaN anyway)
    init=Factotum.NaNStatistics.nanmedian,  # Use median for initial imputation
    maxiter=2000,         # Maximum iterations
    tol=1e-10             # Convergence tolerance
)
nothing # hide
```

### How the EM Algorithm Works

The EM algorithm alternates between:

1. **E-step**: Impute missing values using current factor estimates
   ```math
   \hat{X}_{it} = \hat{F}_t' \hat{\lambda}_i \quad \text{for missing } X_{it}
   ```

2. **M-step**: Re-estimate factors via PCA on the completed data

The algorithm iterates until the maximum change in imputed values falls below the tolerance.

## Practical Example: Macroeconomic Factor Analysis

Let's walk through a complete analysis workflow using the macroeconomic data:

```@example tutorial
# 1. Start with the balanced panel (no missing values)
println("Balanced panel dimensions: $(size(X))")

# 2. Determine the number of factors using multiple criteria
kmax = 10
fm_test = FactorModel(X, kmax; scale=true)
ic_results = informationcriteria((IC1, IC2, IC3, BIC1, BIC2, BIC3), fm_test, kmax)
println("\nOptimal r by different criteria:")
for r in findmin(ic_results)
    println("  ", r)
end

# 3. Fit final model with selected number of factors
r_selected = numfactors(IC1(fm_test, kmax))
fm_final = FactorModel(X, r_selected; scale=true)
Factotum.describe(fm_final)

# 4. Examine factor loadings to interpret factors
Lambda_final = loadings(fm_final)
println("\nTop 5 variables by absolute loading on each factor:")
for k in 1:r_selected
    abs_loadings = abs.(Lambda_final[:, k])
    top_idx = sortperm(abs_loadings, rev=true)[1:5]
    println("Factor $k: ", join(varnames_complete[top_idx], ", "))
end
```

## Tips and Best Practices

1. **Scaling**: Use `scale=true` when variables have different units or variances
2. **Number of factors**: Try multiple information criteria; if they disagree substantially, investigate further
3. **Missing data**: The EM algorithm works well for moderate amounts of missing data (< 30%)
4. **Large datasets**: For very large panels, consider starting with fewer factors and increasing gradually
5. **Interpretation**: Factor loadings indicate which variables load on each factor; use `loadings(fm)` to examine patterns
