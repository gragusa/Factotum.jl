"""
Benchmarks for Factotum.jl

Benchmark suite measuring:
1. PCA factor extraction
2. EM algorithm with missing data
3. Iterative LS with constraints
4. Information criteria computation
"""

using BenchmarkTools
using LinearAlgebra
using Random
using StableRNGs

using Factotum

# ============================================================================
# Data Generation
# ============================================================================

const DEFAULT_SEED = 20240612

function generate_factor_data(rng::AbstractRNG, T::Int, n::Int, r::Int)
    # True factors (T x r)
    F = randn(rng, T, r)

    # True loadings (n x r)
    Lambda = randn(rng, n, r)

    # Noise
    epsilon = 0.5 * randn(rng, T, n)

    # Observed data
    X = F * Lambda' + epsilon

    return X
end

function generate_factor_data_missing(
        rng::AbstractRNG, T::Int, n::Int, r::Int, missing_frac::Float64)
    X = generate_factor_data(rng, T, n, r)

    # Introduce missing values
    n_missing = round(Int, T * n * missing_frac)
    missing_idx = randperm(rng, T * n)[1:n_missing]
    X[missing_idx] .= NaN

    return X
end

# ============================================================================
# Benchmark Suite
# ============================================================================

const SUITE = BenchmarkGroup()

# ----------------------------------------------------------------------------
# PCA Estimation Benchmarks
# ----------------------------------------------------------------------------

SUITE["pca"] = BenchmarkGroup()

# Small dataset
let rng = StableRNG(DEFAULT_SEED)
    X_small = generate_factor_data(rng, 200, 50, 3)

    SUITE["pca"]["T200_n50_r3"] = @benchmarkable FactorModel($X_small, 3; method = :pca)
end

# Medium dataset
let rng = StableRNG(DEFAULT_SEED + 1)
    X_medium = generate_factor_data(rng, 500, 100, 5)

    SUITE["pca"]["T500_n100_r5"] = @benchmarkable FactorModel($X_medium, 5; method = :pca)
end

# Large dataset
let rng = StableRNG(DEFAULT_SEED + 2)
    X_large = generate_factor_data(rng, 1000, 200, 5)

    SUITE["pca"]["T1000_n200_r5"] = @benchmarkable FactorModel($X_large, 5; method = :pca)
end

# Very large dataset
let rng = StableRNG(DEFAULT_SEED + 3)
    X_vlarge = generate_factor_data(rng, 500, 500, 10)

    SUITE["pca"]["T500_n500_r10"] = @benchmarkable FactorModel($X_vlarge, 10; method = :pca)
end

# ----------------------------------------------------------------------------
# EM Algorithm Benchmarks (with missing data)
# ----------------------------------------------------------------------------

SUITE["em"] = BenchmarkGroup()

# Small dataset with 5% missing
let rng = StableRNG(DEFAULT_SEED + 10)
    X = generate_factor_data_missing(rng, 200, 50, 3, 0.05)

    SUITE["em"]["T200_n50_r3_miss5pct"] = @benchmarkable FactorModel($X, 3; method = :em, maxiter = 100)
end

# Medium dataset with 10% missing
let rng = StableRNG(DEFAULT_SEED + 11)
    X = generate_factor_data_missing(rng, 300, 80, 5, 0.10)

    SUITE["em"]["T300_n80_r5_miss10pct"] = @benchmarkable FactorModel($X, 5; method = :em, maxiter = 100)
end

# Larger dataset with 5% missing
let rng = StableRNG(DEFAULT_SEED + 12)
    X = generate_factor_data_missing(rng, 500, 100, 5, 0.05)

    SUITE["em"]["T500_n100_r5_miss5pct"] = @benchmarkable FactorModel($X, 5; method = :em, maxiter = 100)
end

# ----------------------------------------------------------------------------
# Iterative Least Squares Benchmarks (with constraints)
# ----------------------------------------------------------------------------

SUITE["ls"] = BenchmarkGroup()

# Small with constraints
let rng = StableRNG(DEFAULT_SEED + 20)
    X = generate_factor_data(rng, 200, 50, 3)

    # Create simple constraints: normalize first loading on each factor
    c1 = normalize_loading(1, 1; value = 1.0)
    c2 = normalize_loading(2, 2; value = 1.0)
    c3 = normalize_loading(3, 3; value = 1.0)
    constraints = vcat(c1, c2, c3)

    SUITE["ls"]["T200_n50_r3_constrained"] = @benchmarkable FactorModel(
        $X, 3; constraints = $constraints, maxiter = 100)
end

# LS without constraints (for comparison with PCA)
let rng = StableRNG(DEFAULT_SEED + 21)
    X = generate_factor_data(rng, 300, 80, 5)

    SUITE["ls"]["T300_n80_r5_unconstrained"] = @benchmarkable FactorModel($X, 5; method = :ls, maxiter = 100)
end

# LS with missing data and constraints
let rng = StableRNG(DEFAULT_SEED + 22)
    X = generate_factor_data_missing(rng, 200, 50, 3, 0.05)

    c1 = normalize_loading(1, 1; value = 1.0)
    c2 = normalize_loading(2, 2; value = 1.0)
    c3 = normalize_loading(3, 3; value = 1.0)
    constraints = vcat(c1, c2, c3)

    SUITE["ls"]["T200_n50_r3_miss_constrained"] = @benchmarkable FactorModel(
        $X, 3; constraints = $constraints, maxiter = 100)
end

# ----------------------------------------------------------------------------
# Information Criteria Benchmarks
# ----------------------------------------------------------------------------

SUITE["ic"] = BenchmarkGroup()

let rng = StableRNG(DEFAULT_SEED + 30)
    X = generate_factor_data(rng, 200, 50, 3)
    fm = FactorModel(X, 10; method = :pca)  # Estimate with max factors

    SUITE["ic"]["IC1_kmax10"] = @benchmarkable IC1($fm, 10)
    SUITE["ic"]["IC2_kmax10"] = @benchmarkable IC2($fm, 10)
    SUITE["ic"]["IC3_kmax10"] = @benchmarkable IC3($fm, 10)
end

let rng = StableRNG(DEFAULT_SEED + 31)
    X = generate_factor_data(rng, 300, 100, 5)
    fm = FactorModel(X, 15; method = :pca)

    SUITE["ic"]["multiple_criteria_kmax15"] = @benchmarkable informationcriteria(
        (IC1, IC2, IC3, BIC1, BIC2), $fm, 15)
end

# ----------------------------------------------------------------------------
# Factor/Loading Accessors Benchmarks
# ----------------------------------------------------------------------------

SUITE["accessors"] = BenchmarkGroup()

let rng = StableRNG(DEFAULT_SEED + 40)
    X = generate_factor_data(rng, 500, 200, 10)
    fm = FactorModel(X, 10; method = :pca)

    SUITE["accessors"]["factors"] = @benchmarkable factors($fm)
    SUITE["accessors"]["loadings"] = @benchmarkable loadings($fm)
    SUITE["accessors"]["loadings_original"] = @benchmarkable loadings($fm; original_units = true)
    SUITE["accessors"]["explained_variance"] = @benchmarkable explained_variance($fm)
end
