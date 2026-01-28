module Factotum

using LinearAlgebra
using NaNStatistics
using PrettyTables
using Printf
using Statistics
using StatsBase

abstract type AbstractFactorModel end

## ------------------------------------------------------------
## Estimation Method Types
## ------------------------------------------------------------

"""
    AbstractEstimationMethod

Abstract supertype for factor model estimation methods.

Subtypes: [`PCA`](@ref), [`EM`](@ref), [`LeastSquares`](@ref)
"""
abstract type AbstractEstimationMethod end

"""
    PCA <: AbstractEstimationMethod

Principal Component Analysis estimation method.

Used when data has no missing values and no constraints are specified.
This is the default method for complete data.
"""
struct PCA <: AbstractEstimationMethod end

"""
    EM <: AbstractEstimationMethod

Expectation-Maximization estimation method.

Handles missing values (NaN) via iterative imputation. Cannot handle
loading constraints - use [`LeastSquares`](@ref) for constrained estimation.
"""
struct EM <: AbstractEstimationMethod end

"""
    LeastSquares <: AbstractEstimationMethod

Iterative Least Squares estimation method.

Handles both missing values and loading constraints. This is the most
flexible method but may be slower than PCA for complete data.
"""
struct LeastSquares <: AbstractEstimationMethod end

"""
    EstimationStats{V}

Statistics from factor model estimation.

# Fields
- `tss::Float64`: Total sum of squares (standardized data)
- `ssr::Float64`: Sum of squared residuals (standardized data)
- `r2vec::V`: R² for each series
- `nobs::Int`: Number of non-missing observations used for estimation
"""
struct EstimationStats{V <: AbstractVector}
    tss::Float64
    ssr::Float64
    r2vec::V
    nobs::Int
end

struct FactorModel{E <: AbstractEstimationMethod, M <: AbstractMatrix,
    V <: AbstractVector, S <: EstimationStats} <: AbstractFactorModel
    "The matrix of factors"
    factors::M
    "The matrix of loadings"
    loadings::M
    "The eigenvalues of X'X"
    eigenvalues::V
    "Residuals from the factor model (X̄ - F*Λ')"
    residuals::M
    "To demean and scale the matrix X"
    center::M
    scale::M
    "The original matrix"
    X::M
    "The rescaled matrix"
    X̄::M
    "Estimation statistics"
    stats::S
end

struct FactorModelView{M <: AbstractMatrix, S <: AbstractMatrix, V <: AbstractVector, R} <:
       AbstractFactorModel
    "The matrix of factors"
    factors::M
    "The matrix of loadings"
    loadings::S
    "The eigenvalues of X'X"
    eigenvalues::V
    "Residuals from the factor model (X̄ - F*Λ')"
    residuals::R
    "The rescaled matrix"
    X̄::R
end

## ------------------------------------------------------------
## Loading Constraints for Iterative LS Algorithm
## (Defined early so FactorModel constructor can reference it)
## ------------------------------------------------------------

"""
    LoadingConstraints

Linear constraints on factor loadings of the form R × λᵢ = r.

Each constraint specifies:
- Which series (row of Λ) to constrain
- A linear combination of loadings (R vector)
- The value that combination should equal (r)

# Constructors

    LoadingConstraints(series::Vector{Int}, R::Matrix{Float64}, r::Vector{Float64})
    LoadingConstraints(constraints::Matrix)

For the matrix constructor, each row is: [series_index, R₁, R₂, ..., Rₖ, r]

# Example
```julia
# 3-factor model constraints:
# - Series 1: λ₁ = 1 (normalize first loading on factor 1)
# - Series 5: λ₃ = 0 (no loading on factor 3)
constraints = LoadingConstraints(
    [1, 5],                      # series indices
    [1.0 0.0 0.0; 0.0 0.0 1.0],  # R matrices (one row per constraint)
    [1.0, 0.0]                   # r values
)

# Or using matrix format:
constraints = LoadingConstraints([
    1.0  1.0  0.0  0.0  1.0;  # 1*λ₁ + 0*λ₂ + 0*λ₃ = 1
    5.0  0.0  0.0  1.0  0.0;  # 0*λ₁ + 0*λ₂ + 1*λ₃ = 0
])
```
"""
struct LoadingConstraints
    series::Vector{Int}
    R::Matrix{Float64}
    r::Vector{Float64}
end

function LoadingConstraints(constraints::Matrix)
    nc = size(constraints, 1)
    series = Int.(constraints[:, 1])
    R = constraints[:, 2:(end - 1)]
    r = vec(constraints[:, end])
    LoadingConstraints(series, R, r)
end

function Base.vcat(c1::LoadingConstraints, c2::LoadingConstraints)
    LoadingConstraints(
        vcat(c1.series, c2.series),
        vcat(c1.R, c2.R),
        vcat(c1.r, c2.r)
    )
end

"""
    normalize_loading(series, factor, numfactors; value=1.0)

Create a constraint to set the loading of `series` on `factor` to `value`.

# Example
```julia
# Force series 1 to have loading = 1.0 on factor 1 (sign normalization)
c = normalize_loading(1, 1, 3; value=1.0)
```
"""
function normalize_loading(series::Int, factor::Int, numfactors::Int; value::Float64 = 1.0)
    R = zeros(1, numfactors)
    R[1, factor] = 1.0
    LoadingConstraints([series], R, [value])
end

"""
    zero_loading(series, factor, numfactors)

Create a constraint to set the loading of `series` on `factor` to zero.

# Example
```julia
# Force series 5 to have zero loading on factor 3
c = zero_loading(5, 3, 3)
```
"""
function zero_loading(series::Int, factor::Int, numfactors::Int)
    normalize_loading(series, factor, numfactors; value = 0.0)
end

function FactorModel(Z::AbstractMatrix{G}; kwargs...) where {G}
    FactorModel(Z, size(Z, 2); kwargs...)
end

"""
    FactorModel(Z, numfactors; demean=true, scale=false, corrected=false,
                init=nanmean, maxiter=1000, tol=1e-8,
                constraints=nothing, method=:auto, nt_min=10,
                orthonormalize=true)

Estimate a factor model using PCA, EM, or iterative least squares.

Returns a `FactorModel{E}` where `E` is the estimation method type
([`PCA`](@ref), [`EM`](@ref), or [`LeastSquares`](@ref)).

# Arguments
- `Z`: T×n data matrix
- `numfactors`: number of factors to extract
- `demean`: center columns by their means
- `scale`: standardize columns by their std
- `corrected`: use corrected sample std
- `init`: function to compute initial fill values for missing data (nanmean or nanmedian)
- `maxiter`: maximum iterations for EM or LS algorithm
- `tol`: convergence tolerance for EM or LS algorithm
- `constraints`: optional `LoadingConstraints` for restricted estimation (requires `:ls` method)
- `method`: estimation method - `:auto` (default), `:pca`, `:em`, or `:ls`
- `nt_min`: minimum observations per series for LS algorithm (default: 10)
- `orthonormalize`: for `:ls` method, orthonormalize loadings via QR (default: true)

# Methods
- `:pca`: Standard PCA (requires no missing values, no constraints)
- `:em`: EM algorithm (handles missing values, no constraints)
- `:ls`: Iterative least squares (handles missing values and constraints)
- `:auto`: Automatically selects method based on data and constraints:
  - If `constraints` provided → `:ls`
  - If missing values (NaN) detected → `:em`
  - Otherwise → `:pca`

# Querying the estimation method
Use [`estimationmethod`](@ref) to query which method was used:
```julia
fm = FactorModel(X, 3)
estimationmethod(fm)  # returns PCA(), EM(), or LeastSquares()
```

# Example with constraints
```julia
# Constrain series 1 to have loading = 1.0 on factor 1
c = normalize_loading(1, 1, 3; value=1.0)
fm = FactorModel(X, 3; constraints=c)
estimationmethod(fm)  # returns LeastSquares()
```
"""
function FactorModel(Z::AbstractMatrix{G}, numfactors;
        demean::Bool = true, scale::Bool = false, corrected::Bool = false,
        init = nanmean, maxiter::Int = 1000, tol::Float64 = 1e-8,
        constraints::Union{Nothing, LoadingConstraints} = nothing,
        method::Symbol = :auto, nt_min::Int = 10,
        orthonormalize::Bool = true) where {G}
    T, n = size(Z)
    T == 0 && throw(ArgumentError("Input matrix must not be empty (got size $(size(Z)))"))
    numfactors < 0 &&
        throw(ArgumentError("numfactors must be non-negative (got $numfactors)"))
    numfactors > n &&
        throw(ArgumentError("numfactors ($numfactors) must not exceed number of columns ($n)"))

    # Auto-detect missing values
    has_missing = any(isnan, Z)
    has_constraints = constraints !== nothing

    # Validate constraints
    if has_constraints
        max_series = maximum(constraints.series)
        max_series > n &&
            throw(ArgumentError("Constraint references series $max_series but data has only $n columns"))
        size(constraints.R, 2) != numfactors && throw(ArgumentError(
            "Constraint R matrix has $(size(constraints.R, 2)) columns but numfactors=$numfactors"))
    end

    # Determine method
    actual_method = if method == :auto
        if has_constraints
            :ls  # Constraints require iterative LS
        elseif has_missing
            :em  # Missing data requires EM or LS
        else
            :pca
        end
    else
        method
    end

    # Validate method choice
    if actual_method == :pca
        has_missing &&
            throw(ArgumentError("PCA method cannot handle missing values. Use method=:em or method=:ls"))
        has_constraints &&
            throw(ArgumentError("PCA method does not support constraints. Use method=:ls"))
    elseif actual_method == :em
        has_constraints &&
            throw(ArgumentError("EM method does not support constraints. Use method=:ls"))
    end

    # Estimate and construct with appropriate type parameter
    if actual_method == :pca
        (F,
            Λ,
            λ,
            ε,
            μ,
            σₓ,
            Z,
            X,
            stats) = extract_pca(Z, numfactors;
            demean = demean, scale = scale, corrected = corrected)
        FactorModel{PCA, typeof(F), typeof(λ), typeof(stats)}(
            F, Λ, λ, ε, μ, σₓ, Z, X, stats)
    elseif actual_method == :em
        (F,
            Λ,
            λ,
            ε,
            μ,
            σₓ,
            Z,
            X,
            stats) = extract_em(Z, numfactors;
            demean = demean, scale = scale, corrected = corrected,
            init = init, maxiter = maxiter, tol = tol)
        FactorModel{EM, typeof(F), typeof(λ), typeof(stats)}(F, Λ, λ, ε, μ, σₓ, Z, X, stats)
    elseif actual_method == :ls
        (F,
            Λ,
            λ,
            ε,
            μ,
            σₓ,
            Z,
            X,
            stats) = extract_ls(Z, numfactors;
            constraints = constraints, demean = demean, scale = scale,
            nt_min = nt_min, tol = tol, maxiter = maxiter,
            orthonormalize = orthonormalize)
        FactorModel{LeastSquares, typeof(F), typeof(λ), typeof(stats)}(
            F, Λ, λ, ε, μ, σₓ, Z, X, stats)
    else
        throw(ArgumentError("Unknown method: $method. Use :auto, :pca, :em, or :ls"))
    end
end

function extract_pca(
        Z, numfactors; demean::Bool = true, scale::Bool = false, corrected::Bool = false)
    T, n = size(Z)
    μ = demean ? mean(Z; dims = 1) : zeros(1, n)
    σₓ = scale ? std(Z; dims = 1, corrected = corrected) : ones(1, n)
    X = (Z .- μ) ./ σₓ

    (F, Λ, λ) = _pca(X, numfactors)
    ε = X .- F * Λ'

    # Compute estimation statistics
    tss = sum(abs2, X)
    ssr = sum(abs2, ε)
    nobs = T * n

    # R² for each series
    r2vec = Vector{Float64}(undef, n)
    for i in 1:n
        tss_i = sum(abs2, @view X[:, i])
        ssr_i = sum(abs2, @view ε[:, i])
        r2vec[i] = 1.0 - ssr_i / tss_i
    end

    stats = EstimationStats(tss, ssr, r2vec, nobs)
    (F, Λ, λ, ε, μ, σₓ, Z, X, stats)
end

function _pca(X, numfactors)
    T, n = size(X)
    if T > n
        # X'X eigendecomposition
        ev = eigen(Symmetric(X' * X), (n - numfactors + 1):n)
        neg = findall(x -> x < 0, ev.values)
        if !isempty(neg)
            if any(ev.values[neg] .< -9 * eps(Float64) * first(ev.values))
                error("covariance matrix is not non-negative definite")
            else
                ev.values[neg] .= 0.0
            end
        end
        # Eigenvalues of cov(X, corrected=false) = X'X / T
        λ = ev.values[numfactors:-1:1] / T
        # Normalization: Λ'Λ = I (loadings are orthonormal)
        Λ = ev.vectors[:, numfactors:-1:1]
        F = X * Λ
    else
        # XX' eigendecomposition
        ev = eigen(Symmetric(X * X'), (T - numfactors + 1):T)
        neg = findall(x -> x < 0, ev.values)
        if !isempty(neg)
            if any(ev.values[neg] .< -9 * eps(Float64) * first(ev.values))
                error("covariance matrix is not non-negative definite")
            else
                ev.values[neg] .= 0.0
            end
        end
        λ_raw = ev.values[numfactors:-1:1]
        # Eigenvectors of XX' give F_raw (orthonormal columns)
        F_raw = ev.vectors[:, numfactors:-1:1]
        # Λ_raw = X' * F_raw has columns with norm sqrt(λ_raw_i)
        Λ_raw = X' * F_raw
        # Normalization: Λ'Λ = I (loadings are orthonormal)
        # Divide each column by sqrt(λ_raw_i) to normalize
        sqrt_λ_raw = sqrt.(λ_raw)
        Λ = Λ_raw ./ sqrt_λ_raw'
        # Scale F accordingly: F = F_raw * diag(sqrt(λ_raw))
        F = F_raw .* sqrt_λ_raw'
        # Eigenvalues of cov(X, corrected=false) = X'X / T
        λ = λ_raw / T
    end
    (F, Λ, λ)
end

## ------------------------------------------------------------
## EM algorithm for missing data
## ------------------------------------------------------------

"""
    extract_em(Z, numfactors; demean=true, scale=false, corrected=false,
               init=nanmean, maxiter=1000, tol=1e-8)

Estimate factor model using EM algorithm to handle missing values (NaN).

The EM algorithm alternates between:
- E-step: Impute missing values using current factor estimates (Y_ij = F_i * λ_j')
- M-step: Re-estimate factors via PCA on completed data

# Arguments
- `Z`: T×n data matrix (may contain NaN for missing values)
- `numfactors`: number of factors to extract
- `demean`: center columns (using available data)
- `scale`: standardize columns (using available data)
- `corrected`: use corrected sample std
- `init`: function to compute initial fill values (default: nanmean, can use nanmedian)
- `maxiter`: maximum EM iterations
- `tol`: convergence tolerance (max absolute change in imputed values)

# Returns
Same tuple as extract_ΛΛ/extract_FF: (F, Λ, λ, ε, μ, σₓ, Z, X)
"""
function extract_em(Z, numfactors;
        demean::Bool = true, scale::Bool = false, corrected::Bool = false,
        init = nanmean, maxiter::Int = 1000, tol::Float64 = 1e-8)
    T, n = size(Z)

    # 1. Identify missing values
    missing_mask = isnan.(Z)
    has_missing = any(missing_mask)
    missing_indices = findall(missing_mask)

    # 2. Compute statistics on available data (using NaNStatistics)
    μ = demean ? nanmean(Z; dims = 1) : zeros(1, n)
    σₓ = scale ? nanstd(Z; dims = 1, corrected = corrected) : ones(1, n)

    # 3. Center and scale, then do initial imputation (column-wise)
    X = (Z .- μ) ./ σₓ

    if has_missing
        for j in 1:n
            col_mask = @view missing_mask[:, j]
            if any(col_mask)
                col_data = @view X[.!col_mask, j]
                fill_val = isempty(col_data) ? zero(eltype(X)) : init(col_data)
                X[col_mask, j] .= fill_val
            end
        end

        # 4. EM iteration
        converged = false
        max_change = zero(eltype(X))

        for iter in 1:maxiter
            # M-step: PCA on completed data
            (F, Λ, λ) = _pca(X, numfactors)

            # E-step: Impute missing values
            max_change = zero(eltype(X))

            for idx in missing_indices
                i, j = idx[1], idx[2]
                pred = dot(view(F, i, :), view(Λ, j, :))

                change = abs(pred - X[idx])
                max_change = max(max_change, change)
                X[idx] = pred
            end

            if max_change < tol
                converged = true
                break
            end
        end

        if !converged
            @warn "EM algorithm did not converge after $maxiter iterations (final change: $(max_change))"
        end
    end

    # 5. Final PCA
    (F, Λ, λ) = _pca(X, numfactors)
    ε = X .- F * Λ'

    # Compute estimation statistics (on standardized, imputed data)
    tss = sum(abs2, X)
    ssr = sum(abs2, ε)

    # nobs = number of non-missing observations in original data
    nobs = count(!isnan, Z)

    # R² for each series
    r2vec = Vector{Float64}(undef, n)
    for i in 1:n
        tss_i = sum(abs2, @view X[:, i])
        ssr_i = sum(abs2, @view ε[:, i])
        r2vec[i] = 1.0 - ssr_i / tss_i
    end

    stats = EstimationStats(tss, ssr, r2vec, nobs)
    (F, Λ, λ, ε, μ, σₓ, Z, X, stats)
end

## ------------------------------------------------------------
## Iterative LS Algorithm with Constraints
## ------------------------------------------------------------

"""
    extract_ls(Z, numfactors; constraints=nothing, demean=true, scale=false,
               nt_min=10, tol=1e-8, maxiter=1000)

Estimate factor model using iterative least squares (MATLAB-style algorithm).
Supports linear constraints on loadings and handles missing data.

The algorithm alternates between:
1. Update loadings: For each series, regress data on factors (with constraints if specified)
2. Update factors: For each time period, regress data on loadings

# Arguments
- `Z`: T×n data matrix (may contain NaN for missing values)
- `numfactors`: number of factors to extract
- `constraints`: optional `LoadingConstraints` for restricted estimation
- `demean`: center columns by their means
- `scale`: standardize columns by their std (uses population std, i.e., N divisor)
- `nt_min`: minimum number of observations for a series to estimate its loadings
- `tol`: convergence tolerance (change in SSR relative to T*n)
- `maxiter`: maximum iterations

# Returns
Same tuple as other extract functions: (F, Λ, λ, ε, μ, σₓ, Z, X, stats)

# Notes
This implementation matches the MATLAB factor_estimation_ls function:
- Uses population standard deviation (N divisor) for standardization
- Initializes via PCA on balanced panel (rows with no missing data)
- Computes R² for each series by re-regressing on final factors
"""
function extract_ls(Z, numfactors;
        constraints::Union{Nothing, LoadingConstraints} = nothing,
        demean::Bool = true, scale::Bool = false,
        nt_min::Int = 10, tol::Float64 = 1e-8, maxiter::Int = 1000,
        orthonormalize::Bool = true)
    T, n = size(Z)

    # Standardize using available data (MATLAB uses population std)
    μ = demean ? nanmean(Z; dims = 1) : zeros(1, n)
    # MATLAB: xstd = (nanstd(xdata).*mult)' where mult adjusts to population std
    σₓ = scale ? nanstd(Z; dims = 1, corrected = false) : ones(1, n)
    X = (Z .- μ) ./ σₓ

    # Scale constraint values if data is scaled
    constraints_scaled = constraints
    if constraints !== nothing && scale
        # Adjust r values for standardization: r_scaled = r / σ for the constrained series
        r_scaled = copy(constraints.r)
        for (idx, i) in enumerate(constraints.series)
            r_scaled[idx] = constraints.r[idx] / σₓ[i]
        end
        constraints_scaled = LoadingConstraints(constraints.series, constraints.R, r_scaled)
    end

    # Compute total sum of squares and nobs (on standardized data, ignoring NaN)
    tss = zero(eltype(X))
    nobs = 0
    for i in 1:n
        for t in 1:T
            val = X[t, i]
            if !isnan(val)
                tss += val^2
                nobs += 1
            end
        end
    end

    # Initialize factors via PCA on balanced panel (columns with no missing data)
    # MATLAB: xbal = packr(xdata_std')' removes columns with NaN
    balanced_mask = vec(.!any(isnan.(X), dims = 1))
    n_complete = sum(balanced_mask)

    if n_complete >= numfactors
        # Enough complete columns: use balanced panel PCA
        X_bal = X[:, balanced_mask]
        (F, Λ_bal, _) = _pca(X_bal, numfactors)

        # Initialize full Λ matrix - loadings for complete columns come from PCA
        Λ_init = fill(NaN, n, numfactors)
        Λ_init[balanced_mask, :] = Λ_bal

        # For columns with missing data, initialize loadings by regressing on factors
        for i in findall(.!balanced_mask)
            xᵢ = @view X[:, i]
            valid = .!isnan.(xᵢ)
            if sum(valid) >= numfactors
                F_valid = F[valid, :]
                x_valid = xᵢ[valid]
                Λ_init[i, :] = F_valid \ x_valid
            end
        end
    else
        # Not enough complete columns: use EM-style initialization
        X_imputed = copy(X)
        for j in 1:n
            col = @view X_imputed[:, j]
            col_missing = isnan.(col)
            if any(col_missing)
                col_mean = mean(col[.!col_missing])
                X_imputed[col_missing, j] .= isnan(col_mean) ? zero(eltype(X)) : col_mean
            end
        end
        (F, Λ_init, _) = _pca(X_imputed, numfactors)
    end

    # Use initialized loadings (will be updated in first iteration)
    Λ = Λ_init

    # Iterative LS algorithm
    ssr_old = Inf
    ssr = zero(eltype(X))
    converged = false

    for iter in 1:maxiter
        # Step 1: Update loadings (series by series)
        for i in 1:n
            xᵢ = @view X[:, i]
            valid = .!isnan.(xᵢ)
            nvalid = sum(valid)

            if nvalid >= nt_min
                F_valid = F[valid, :]
                x_valid = xᵢ[valid]

                # OLS estimate
                FtF_inv = inv(F_valid' * F_valid)
                λ_ols = FtF_inv * (F_valid' * x_valid)

                # Check for constraints on this series
                if constraints_scaled !== nothing
                    constraint_idx = findall(==(i), constraints_scaled.series)
                    if !isempty(constraint_idx)
                        R = constraints_scaled.R[constraint_idx, :]
                        r = constraints_scaled.r[constraint_idx]

                        # Restricted LS: λ_rls = λ_ols - (F'F)⁻¹ R' [R (F'F)⁻¹ R']⁻¹ (R λ_ols - r)
                        tmp1 = FtF_inv * R'
                        tmp2 = inv(R * tmp1)
                        λ_ols = λ_ols - tmp1 * tmp2 * (R * λ_ols - r)
                    end
                end

                Λ[i, :] = λ_ols
            end
        end

        # Step 2: Update factors (time period by time period)
        for t in 1:T
            xₜ = @view X[t, :]
            valid = .!isnan.(xₜ)
            nvalid = sum(valid)

            if nvalid >= numfactors
                Λ_valid = Λ[valid, :]
                x_valid = xₜ[valid]
                F[t, :] = Λ_valid \ x_valid
            end
        end

        # Compute SSR (sum of squared residuals, ignoring NaN)
        ssr = zero(eltype(X))
        for i in 1:n
            for t in 1:T
                val = X[t, i]
                if !isnan(val)
                    pred = dot(@view(F[t, :]), @view(Λ[i, :]))
                    ssr += (val - pred)^2
                end
            end
        end

        # Check convergence (MATLAB: diff = abs(ssr_old - ssr), while diff > tol*(nt*ns))
        if abs(ssr_old - ssr) < tol * T * n
            converged = true
            break
        end
        ssr_old = ssr
    end

    if !converged
        @warn "Iterative LS did not converge after $maxiter iterations"
    end

    # Compute R² for each series (MATLAB re-regresses each series on final factors)
    r2vec = fill(NaN, n)
    for i in 1:n
        xᵢ = @view X[:, i]
        valid = .!isnan.(xᵢ)
        nvalid = sum(valid)

        if nvalid >= nt_min
            F_valid = F[valid, :]
            x_valid = xᵢ[valid]

            # OLS regression for R²
            b = F_valid \ x_valid
            e = x_valid - F_valid * b
            r2_ssr = dot(e, e)
            r2_tss = dot(x_valid, x_valid)
            r2vec[i] = 1.0 - r2_ssr / r2_tss
        end
    end

    # Impute missing values (like EM does) so X̄ has no NaN
    X_imputed = copy(X)
    for i in 1:n
        for t in 1:T
            if isnan(X[t, i])
                X_imputed[t, i] = dot(@view(F[t, :]), @view(Λ[i, :]))
            end
        end
    end

    # Orthonormalize loadings if requested and no constraints (so information criteria work correctly)
    # With constraints, keep the raw LS solution to preserve constraint structure
    if orthonormalize && constraints === nothing
        # Orthonormalize loadings via QR decomposition
        # Λ = Q * R, so F * Λ' = F * R' * Q' = (F * R') * Q'
        # Let Λ_orth = Q (orthonormal), F_orth = F * R'
        qrΛ = qr(Λ)
        Λ_orth = Matrix(qrΛ.Q)  # n × r, orthonormal columns
        F_orth = F * Matrix(qrΛ.R)'  # T × r

        # Eigenvalues from orthonormalized factor variances
        λ = vec(var(F_orth, dims = 1, corrected = false))

        # Store orthonormal loadings (same convention as PCA/EM)
        # loadings(fm; original_units=true) will return Λ .* σₓ
        Λ_final = Λ_orth

        # Compute residuals (on standardized, imputed data)
        ε = X_imputed .- F_orth * Λ_orth'
    else
        # With constraints: keep raw LS solution to preserve constraints
        # Note: Information criteria may not work correctly with constrained LS
        F_orth = F
        λ = vec(var(F, dims = 1, corrected = false))
        Λ_final = Λ  # Store standardized loadings (preserves constraints)
        ε = X_imputed .- F * Λ'
    end

    stats = EstimationStats(tss, ssr, r2vec, nobs)
    (F_orth, Λ_final, λ, ε, μ, σₓ, Z, X_imputed, stats)
end

function Base.view(fm::FactorModel, k::Int)
    k <= 0 &&
        throw(ArgumentError("Cannot view a FactorModel with $k factors (must be positive)"))
    view(fm, 1:k)
end

function Base.view(fm::FactorModel, rnge::UnitRange)
    isempty(rnge) && throw(ArgumentError("Range must not be empty"))
    first(rnge) <= 0 &&
        throw(ArgumentError("Range must start at 1 or greater (got $(first(rnge)))"))
    maximum(rnge) > numfactors(fm) && throw(ArgumentError(
        "Cannot create FactorModel view with $(maximum(rnge)) factors when parent has $(numfactors(fm)) factors"))
    FactorModelView(view(factors(fm), :, rnge), view(loadings(fm), :, rnge),
        view(eigvals(fm), rnge), residuals(fm), fm.X̄)
end

## ------------------------------------------------------------
## Methods
## ------------------------------------------------------------

Base.size(fm::AbstractFactorModel) = size(fm.X̄)

"""
    numfactors(fm::AbstractFactorModel)

Return the number of factors in the factor model.

# Example
```julia
fm = FactorModel(X, 5)
numfactors(fm)  # returns 5
```
"""
numfactors(fm::AbstractFactorModel) = size(loadings(fm), 2)

"""
    loadings(fm::FactorModel; original_units::Bool=false)

Return the n×r matrix of factor loadings, where n is the number of variables
and r is the number of factors.

# Arguments
- `original_units`: If `true` and the model was fit with `scale=true`, return
  loadings scaled to original data units (multiplied by column standard deviations).
  Default is `false`.

# Example
```julia
fm = FactorModel(X, 3; scale=true)
Λ = loadings(fm)                      # n×3 matrix (standardized units)
Λ_orig = loadings(fm; original_units=true)  # n×3 matrix (original units)
```
"""
function loadings(fm::FactorModel; original_units::Bool = false)
    Λ = fm.loadings
    if original_units
        return Λ .* vec(fm.scale)
    end
    return Λ
end

loadings(fmv::FactorModelView) = fmv.loadings

"""
    factors(fm::AbstractFactorModel)

Return the T×r matrix of estimated factors, where T is the number of observations
and r is the number of factors.

# Example
```julia
fm = FactorModel(X, 3)
F = factors(fm)  # T×3 matrix
```
"""
factors(fm::AbstractFactorModel) = fm.factors

LinearAlgebra.eigvals(fm::AbstractFactorModel) = fm.eigenvalues

function sdev(fm::AbstractFactorModel)
    λ = eigvals(fm)
    sqrt.(λ)
end

"""
    explained_variance(fm::FactorModel)

Return a vector of proportions of variance explained by each factor,
relative to the total variance in the data.

The sum of all proportions equals the fraction of total variance captured
by the r extracted factors (typically < 1).

# Example
```julia
fm = FactorModel(X, 3)
ev = explained_variance(fm)
sum(ev)  # fraction of total variance explained by 3 factors
```
"""
function explained_variance(fm::FactorModel)
    T, _ = size(fm)
    λ = eigvals(fm)
    # Total variance = sum of all eigenvalues of cov(X) = sum(abs2, X̄) / T
    total_var = sum(abs2, fm.X̄) / T
    λ ./ total_var
end

function StatsBase.residuals(fm::AbstractFactorModel)
    F = factors(fm)
    Λ = loadings(fm)
    fm.residuals .= fm.X̄ .- F*Λ'
    return fm.residuals
end

X(fm::FactorModel) = fm.X
X̄(fm::AbstractFactorModel) = fm.X̄

"""
    stats(fm::FactorModel)

Return the `EstimationStats` struct containing estimation statistics.

# Example
```julia
fm = FactorModel(X, 3)
s = stats(fm)
s.tss   # total sum of squares
s.ssr   # sum of squared residuals
s.r2vec # R² for each series
s.nobs  # number of observations
```
"""
stats(fm::FactorModel) = fm.stats

"""
    tss(fm::FactorModel)

Return the total sum of squares (on standardized data).
"""
tss(fm::FactorModel) = fm.stats.tss

"""
    ssr(fm::FactorModel)

Return the sum of squared residuals (on standardized data).
"""
ssr(fm::FactorModel) = fm.stats.ssr

"""
    r2(fm::FactorModel)

Return a vector of R² values for each series.
"""
r2(fm::FactorModel) = fm.stats.r2vec

"""
    nobs(fm::FactorModel)

Return the number of non-missing observations used for estimation.
"""
nobs(fm::FactorModel) = fm.stats.nobs

"""
    estimationmethod(fm::FactorModel)

Return the estimation method used to fit the factor model.

Returns one of: `PCA()`, `EM()`, or `LeastSquares()`.

# Example
```julia
fm = FactorModel(X, 3)
estimationmethod(fm)  # returns PCA()

fm_em = FactorModel(X_with_nans, 3)
estimationmethod(fm_em)  # returns EM()

fm_ls = FactorModel(X, 3; method=:ls)
estimationmethod(fm_ls)  # returns LeastSquares()
```
"""
estimationmethod(::FactorModel{E}) where {E} = E()

## Output

# Helper functions for method names
_method_name(::Type{PCA}) = "Principal Component Analysis"
_method_name(::Type{EM}) = "EM Algorithm"
_method_name(::Type{LeastSquares}) = "Iterative Least Squares"

function Base.show(io::IO, fm::FactorModel{E}) where {E}
    printstyled(io, "\nStatic Factor Model\n", color = :green)
    @printf io "Dimensions of X..........: %s\n" size(fm)
    @printf io "Number of factors........: %s\n" numfactors(fm)
    @printf io "Estimation method........: %s\n" _method_name(E)
end

function Base.show(io::IO, fmv::FactorModelView)
    printstyled(io, "\nStatic Factor Model (View)\n", color = :green)
    @printf io "Dimensions of X..........: %s\n" size(fmv.X̄)
    @printf io "Number of factors........: %s\n" numfactors(fmv)
end

"""
    describe(fm::FactorModel)
    describe(io::IO, fm::FactorModel)

Print a detailed summary of the factor model, including:
- Model dimensions (T × n)
- Number of factors
- Factor importance table with standard deviations, proportion of variance,
  and cumulative proportion

# Example
```julia
fm = FactorModel(X, 3)
describe(fm)
```
"""
describe(fm::FactorModel) = describe(stdout, fm)

function describe(io::IO, fm::FactorModel)
    show(io, fm)
    printstyled(io, "Factors' importance:\n", color = :green)
    factortable(io, fm)
end

function factortable(io::IO, fm::FactorModel)
    k = numfactors(fm)
    explainedvar = explained_variance(fm)
    colnms = "Factor_" .* string.(1:k)
    rownms = ["Standard deviation", "Proportion of Variance", "Cumulative Proportion"]
    mat = Matrix{Float64}(undef, 3, k)
    mat[1, :] .= sdev(fm)
    mat[2, :] .= explainedvar
    cumsum!(view(mat, 3, :), explainedvar)
    ct = CoefTable(mat, colnms, rownms)
    show(io, ct)
end

## ------------------------------------------------------------
## Information Criteria
## ------------------------------------------------------------
abstract type AbstractInformationCriterion end
struct IC1 <: AbstractInformationCriterion end
struct IC2 <: AbstractInformationCriterion end
struct IC3 <: AbstractInformationCriterion end
struct PCp1 <: AbstractInformationCriterion end
struct PCp2 <: AbstractInformationCriterion end
struct PCp3 <: AbstractInformationCriterion end
struct AIC1 <: AbstractInformationCriterion end
struct AIC2 <: AbstractInformationCriterion end
struct AIC3 <: AbstractInformationCriterion end
struct BIC1 <: AbstractInformationCriterion end
struct BIC2 <: AbstractInformationCriterion end
struct BIC3 <: AbstractInformationCriterion end

struct InformationCriterion{M <: AbstractInformationCriterion, T <: AbstractFloat}
    criterion::M
    crit::Array{T, 1}
    rnge::UnitRange{Int64}
end

## Calculate V(F̂ᵏ) for k ⩽ kₘₐₓ
## Uses NaN-aware sum to handle LS method with missing data
function V(fmv::FactorModelView)
    ε = residuals(fmv)
    NaNStatistics.nansum(abs2.(ε)) / count(!isnan, ε)
end

V(fm::FactorModel, kₘₐₓ) = [V(view(fm, j)) for j in 1:kₘₐₓ]

variance_factor(::Type{M}, fm, kₘₐₓ) where {M <: Union{IC1, IC2, IC3}} = 1.0
function variance_factor(::Type{M}, fm, kₘₐₓ) where {M <: AbstractInformationCriterion}
    V(view(fm, kₘₐₓ))
end ## This is probably transform
transform_V(::Type{M}, V) where {M <: Union{IC1, IC2, IC3}} = log.(V)
transform_V(::Type{M}, V) where {M <: AbstractInformationCriterion} = V

function informationcriterion(
        s::Type{M}, fm::FactorModel, kₘₐₓ::Int64) where {M <: AbstractInformationCriterion}
    kₘₐₓ <= 0 && throw(ArgumentError("kₘₐₓ must be positive (got $kₘₐₓ)"))
    kₘₐₓ > numfactors(fm) &&
        throw(ArgumentError("kₘₐₓ ($kₘₐₓ) exceeds number of factors in model ($(numfactors(fm)))"))
    T, n = size(fm)
    rnge = 1:kₘₐₓ
    σ̂² = variance_factor(s, fm, kₘₐₓ)
    VV = [NaNStatistics.nansum(abs2.(fm.X̄)) / count(!isnan, fm.X̄); V(fm, kₘₐₓ)]
    Vₖ = transform_V.(s, VV)
    gₜₙ = map(k -> k*penalty(s, T, n, k), 0:last(rnge))
    InformationCriterion(M(), Vₖ + σ̂² .* gₜₙ, 0:last(rnge))
end

function informationcriteria(criterion::Tuple, fm, kₘₐₓ)
    @assert all(map(x->isa(x(), Factotum.AbstractInformationCriterion), criterion)) "Some of the arguments is not a SelectionCriterion"
    map(x->x(fm, kₘₐₓ), criterion)
end

(for criterion in
     (:BIC1, :BIC2, :BIC3, :AIC1, :AIC2, :AIC3, :IC1, :IC2, :IC3, :PCp1, :PCp2, :PCp3)
    eval(quote
        function ($criterion)(fm, kₘₐₓ::Int64)
            Factotum.informationcriterion($criterion, fm, kₘₐₓ)
        end
        Base.string(ic::($criterion)) = string(($criterion))
        function ($criterion)(X::Matrix, kₘₐₓ::Int64; kwargs...)
            Factotum.informationcriterion($criterion, FactorModel(X, kₘₐₓ; kwargs...), kₘₐₓ)
        end
    end)
end)

function Base.findmin(ic::InformationCriterion)
    fmin = findmin(ic.crit)
    NamedTuple{(Symbol(string(ic.criterion)), :r)}(fmin)
end

function Base.findmin(ic::Tuple{Vararg{InformationCriterion, N}}) where {N}
    fmin = map(x->findmin(x.crit), ic)
    nm = map(x->(Symbol(x.criterion), :r), ic)
    TT = map(x->eltype(x.crit), ic)
    map((nm, x, T) -> NamedTuple{nm}(x), nm, fmin, TT)
end

numfactors(ic::InformationCriterion) = findmin(ic).r
Base.string(ic::InformationCriterion{T, F}) where {T, F} = string(T)

"""
    criterion(ic::InformationCriterion)

Return the vector of information criterion values for each number of factors tested
(from 0 to kmax).

# Example
```julia
fm = FactorModel(X, 10)
ic1 = IC1(fm, 10)
values = criterion(ic1)  # Vector of 11 values (for r = 0, 1, ..., 10)
```
"""
criterion(ic::InformationCriterion) = ic.crit

function Base.show(io::IO, ic::T) where {T <: InformationCriterion}
    column_labels = ["# of factors", "Criterion"]
    highlight1 = TextHighlighter((data, i, j) -> data[i, 2] == minimum(data[:, 2]),
        Crayon(background = :blue, foreground = :white, bold = true))
    highlight2 = TextHighlighter((data, i, j) -> j == 2, Crayon(foreground = :light_blue))
    highlight3 = TextHighlighter((data, i, j) -> j == 1, Crayon(foreground = :light_red, bold = true))
    style = TextTableStyle(first_line_column_label = crayon"yellow bold")
    pretty_table(io, [ic.rnge ic.crit];
        column_labels = [column_labels],
        style = style,
        formatters = [fmt__printf("%5.0f", [1]), fmt__printf("%5.3g", [2])],
        highlighters = [highlight1, highlight2, highlight3])
end

function Base.show(io::IO, ic::Tuple{
        InformationCriterion, Vararg{InformationCriterion, N}}) where {N}
    column_labels = ["# of factors", string.(ic)...]
    highlights = [TextHighlighter(
                      (data, i, j) -> data[i, j] == minimum(data[:, x]) && j > 1,
                      Crayon(background = :blue, foreground = :white, bold = true))
                  for x in 2:(length(ic) + 1)]
    tbl = [first(ic).rnge mapreduce(x -> x.crit, hcat, ic)]
    style = TextTableStyle(first_line_column_label = crayon"yellow bold")
    pretty_table(io, tbl;
        column_labels = [column_labels],
        style = style,
        formatters = [
            fmt__printf("%5.0f", [1]), fmt__printf("%5.3g", collect(2:(length(ic) + 1)))],
        highlighters = highlights)
end

function penalty(s::Type{P}, T, N) where {P <: Union{IC1, PCp1}}
    NtT = N*T
    NpT = N+T
    p1 = NpT/NtT
    p2 = log(NtT/NpT)
    p1*p2
end

function penalty(s::Type{P}, T, N) where {P <: Union{IC2, PCp2}}
    C2 = min(T, N)
    NtT = N*T
    NpT = N+T
    p1 = NpT/NtT
    p2 = log(C2)
    p1*p2
end

function penalty(s::Type{P}, T, N) where {P <: Union{IC3, PCp3}}
    C2 = min(T, N)
    log(C2)/C2
end

penalty(s::Type{S}, T, N, k) where {S} = penalty(s, T, N)

penalty(s::Type{AIC1}, T, N) = 2/T
penalty(s::Type{AIC2}, T, N) = 2/N
penalty(s::Type{AIC3}, T, N, k) = 2*(N+T-k)/(N*T)

penalty(s::Type{BIC1}, T, N) = log(T)/T
penalty(s::Type{BIC2}, T, N) = log(N)/N
penalty(s::Type{BIC3}, T, N, k) = ((N+T-k)*log(N*T))/(N*T)

############################################################
## Wald test
############################################################
# struct WaldTest
#     tbl::NamedTuple
#     rankmin::Int64
#     rankₘₐₓ::Int64
# end

# struct WaldTestFun{F, T, Z}
#     f::F
#     r::Int64
#     vecsigma::T
#     Vhat::Z
# end

# (wf::WaldTestFun)(theta) = wf.f(theta, wf.r, wf.vecsigma, wf.Vhat)

# function waldobjfun(th, r, vecsigma, Vhat)
#     ##r,k = size(theta) ## note that the rank being tested is r0 = r-1
#     theta = reshape(th, r+1, length(th)÷(r+1))
#     sigmamat = diagm(0=>theta[1,:].^2) .+ theta[2:r+1,:]'*theta[2:r+1,:]
#     tempsigma = sigmamat[findall(tril(ones(size(sigmamat))).==1)]
#     (vecsigma -tempsigma)' /Vhat *(vecsigma - tempsigma)
# end

# X = randn(100,10);
# fm = Factotum.FactorModel(X)

# function waldtest(fm::FactorModel, minrank::Int = 0, maxrank::Int = 2)
#     X = copy(fm.X)
#     T, n = size(X)
#     ## Normalize factor
#     Xs = X / diagm(0=>sqrt.(diag(cov(X))))
#     covX = cov(Xs)
#     meanX = mean(Xs, dims=1)
#     vecsigma = Factotum.vech(covX)
#     bigN = length(vecsigma)
#     Vhat = Array{Float64}(undef, bigN, bigN)
#     varvecsig = zeros(n,n,n,n);

#     for i1 in 1:n, i2 = 1:n, i3 = 1:n, i4 = 1:n
#         varvecsig[i1,i2,i3,i4] = sum( (Xs[:,i1] .- meanX[i1]).*(Xs[:,i2] .- meanX[i2]).*(Xs[:,i3] .- meanX[i3]) .*(Xs[:,i4] .- meanX[i4])) / T^2 - covX[i1,i2] *covX[i3,i4] /T
#     end

#     idx = findall(tril(ones(size(covX))).==1)
#     for i=1:bigN, j=1:bigN   ## map elements of varvecsig array into matrix corresponding to
#         Vhat[i,j] = varvecsig[idx[i],idx[j]]
#     end

#     out_table = (rank = -1, waldstat = NaN, df = NaN, pvalue = NaN)

#     ## Initial values
#     for k in minrank:maxrank
#         wf = WaldTestFun(waldobjfun, k, vecsigma, Vhat)
#         df = (n-k)*(n-k+1)/2 - n

#         theta0 = theta_initial_value(n,k)

#         outs = Array{Tuple{Float64,Array{Float64,1},Bool},1}()

#         for j in theta0
#             out = Optim.optimize(wf, j, BFGS(), Optim.Options(allow_f_increases=true); autodiff=:forward)
#             push!(outs, (out.minimum::Float64, out.minimizer::Array{Float64,1}, Optim.converged(out)::Bool))
#         end

#         convouts = outs[map(x->x[3], outs)]
#         out      = convouts[argmin(map(x->x[1], convouts))]

#         dfa = (rank = k, waldstat = out[1], df = df,  pvalue = 1-StatsFuns.chisqcdf(df, out[1]))
#         append!(out_table, dfa)
#     end
#     filter!(raw->raw[:rank]>=0, out_table)
#     WaldTest(out_table, minrank, maxrank)
# end

# function theta_initial_value(n,k)
#     I3 = ones(1,n)/3
#     ek = [diagm(0=>ones(k)) zeros(k, n-k)]
#     t0 = ([I3; zeros(k,n)], [I3; ones(k,n)./(2*k)],[I3; ek./(2*k)], [I3; reverse(ek./(2*k), dims=2)])::NTuple{4,Array{Float64,2}}
#     map(vec, t0)::NTuple{4,Array{Float64,1}}
# end

# function vech(X::Matrix{S}) where S
#     T, n = size(X)
#     r = round(Int64, n*(n+1)/2)
#     x = Array{S, 1}(undef, r)
#     i = 1
#     for j in 1:n, k in j:n
#         x[i] = X[j,k]
#         i += 1
#     end
#     x
# end

export FactorModel, EstimationStats, describe,
       numfactors, factors, loadings, explained_variance,
# Estimation method types and accessor
       AbstractEstimationMethod, PCA, EM, LeastSquares, estimationmethod,
# Estimation statistics
       stats, tss, ssr, r2, nobs,
# Information criteria
       IC1, IC2, IC3, PCp1, PCp2, PCp3,
       AIC1, AIC2, AIC3, BIC1, BIC2, BIC3,
       informationcriteria, criterion,
# Constrained factor estimation
       LoadingConstraints, normalize_loading, zero_loading

end # module"
