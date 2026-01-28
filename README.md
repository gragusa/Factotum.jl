# Factotum.jl

[![CI](https://github.com/gragusa/Factotum.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/gragusa/Factotum.jl/actions/workflows/CI.yml) [![codecov.io](http://codecov.io/github/gragusa/Factotum.jl/coverage.svg?branch=master)](http://codecov.io/github/gragusa/Factotum.jl?branch=master) [![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl) ![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826) ![lifecycle](https://img.shields.io/badge/lifecycle-stable-green.svg)

A Julia package for estimating **static factor models**. 

`Factotum.jl` supports multiple estimation methods (PCA, EM, iterative Least Squares), handles missing data, and provides model selection via information criteria.

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
fm = FactorModel(X, 5)

# Access results
F = factors(fm)           # T × r factor matrix
Λ = loadings(fm)          # n × r loadings matrix
ev = explained_variance(fm)

# View model summary
describe(fm)
```

## Features

- **Multiple estimation methods**: PCA (default), EM algorithm, iterative Least Squares
- **Missing data support**: EM and LS methods handle NaN values
- **Constrained estimation**: Linear constraints on factor loadings via LS method (experimental)
- **Model selection**: Bai & Ng (2002) information criteria (IC1-3, PCp1-3, AIC1-3, BIC1-3)

## Estimation Methods

```julia
# PCA (default for complete data)
fm = FactorModel(X, 5)

# EM algorithm (auto-selected for data with NaN)
fm = FactorModel(X_with_missing, 5)

# Iterative Least Squares (required for constraints)
fm = FactorModel(X, 5; method=:ls)

# Query the estimation method used
estimationmethod(fm)  # returns PCA(), EM(), or LeastSquares()
```

## Model Selection

```julia
fm = FactorModel(X, 10)

# Compute information criterion
ic = IC1(fm, 10)

# Optimal number of factors
r_opt = numfactors(ic)

# Compare multiple criteria
ics = informationcriteria((IC1, IC2, BIC3), fm, 10)
```

## Documentation

See the [documentation](https://gragusa.github.io/Factotum.jl/) for detailed usage, tutorials, and API reference.

## References

- Bai, J., & Ng, S. (2002). Determining the number of factors in approximate factor models. *Econometrica*, 70(1), 191-221.

## License

MIT License
