# Factotum.jl

*A Julia package for estimating static factor models using Principal Component Analysis (PCA).*

## Overview

Factotum.jl provides tools for estimating factor models from panel data. The package supports:

- **Principal Component Analysis (PCA)** for extracting latent factors from high-dimensional data
- **EM algorithm** for handling missing values (NaN) in the data
- **Information criteria** (IC, PCp, AIC, BIC variants) for selecting the optimal number of factors
- **Efficient computation** that automatically selects the optimal eigendecomposition strategy based on data dimensions

## Installation

```julia
using Pkg
Pkg.add("Factotum")
```

## Quick Start

We demonstrate Factotum.jl using a panel of 233 quarterly US macroeconomic time series (1959-2024), including output, employment, prices, interest rates, and money supply indicators. The data has been transformed for stationarity and locally demeaned.

```@example quickstart
using Factotum
using CSV, DataFrames

# Load quarterly US macroeconomic data
datapath = joinpath(dirname(dirname(pathof(Factotum))), "test", "data", "macrodata.csv")
df = CSV.read(datapath, DataFrame)
X_full = Matrix{Float64}(df[:, 2:end])  # Exclude DATE column

# Create balanced panel: drop series with any missing values
complete_cols = [all(x -> !isnan(x), col) for col in eachcol(X_full)]
X = X_full[:, complete_cols]

println("Balanced panel: $(size(X, 1)) quarters × $(size(X, 2)) variables")

# Fit a factor model with 5 factors
fm = FactorModel(X, 5; scale=true)

# View model summary
Factotum.describe(fm)
```

Extract the estimated factors and loadings:

```@example quickstart
F = factors(fm)       # T×r factor matrix
Lambda = loadings(fm) # n×r loadings matrix
println("Factor matrix size: ", size(F))
println("Loadings matrix size: ", size(Lambda))
```

## Model Selection

Use information criteria to determine the optimal number of factors:

```@example quickstart
# Fit model with maximum number of factors
fm_max = FactorModel(X, 10; scale=true)

# Compute IC1 criterion for 0 to 10 factors
ic = IC1(fm_max, 10)
println("Optimal number of factors (IC1): ", numfactors(ic))
```

## Handling Missing Data

Factotum.jl automatically handles missing values using an EM algorithm. The full macroeconomic dataset contains series with missing values due to different start dates and data transformations:

```@example quickstart
# Use the full dataset with missing values
println("Full dataset: $(size(X_full, 1)) quarters × $(size(X_full, 2)) variables")
println("Missing values: ", sum(isnan.(X_full)), " (", round(100 * sum(isnan.(X_full)) / length(X_full), digits=1), "%)")

# Fit model - EM algorithm is automatically used
fm_em = FactorModel(X_full, 5; scale=true)
Factotum.describe(fm_em)
```

## Contents

```@contents
Pages = ["tutorial.md", "api.md"]
Depth = 2
```

## Index

```@index
```
