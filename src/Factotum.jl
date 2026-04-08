module Factotum

using LinearAlgebra
using NaNStatistics
using PrettyTables
using Printf
using Statistics
using StatsBase
using Tables

abstract type AbstractFactorModel end

"""
    AbstractInformationCriterion

Abstract supertype for information criteria used to select the number of factors.

All criteria can be called as functions: `IC1(fm, kmax)` or `IC1(X, kmax; kwargs...)`.

Use `numfactors(ic)` to get the optimal r, or `criterion(ic)` for all values.

# Available Criteria

From Bai and Ng (2002):
- `IC1`, `IC2`, `IC3` - Information criteria
- `PCp1`, `PCp2`, `PCp3` - Panel Cp variants

Standard criteria:
- `AIC1`, `AIC2`, `AIC3` - AIC-based
- `BIC1`, `BIC2`, `BIC3` - BIC-based

# Examples

```jldoctest
julia> X = randn(100, 20);

julia> fm = FactorModel(X, 8);

julia> ic = IC1(fm, 8);

julia> numfactors(ic) >= 0
true
```

Direct from data:

```jldoctest
julia> X = randn(100, 20);

julia> ic = IC1(X, 5; scale=true);

julia> numfactors(ic) >= 0
true
```

See also: [`informationcriteria`](@ref), [`criterion`](@ref)
"""
abstract type AbstractInformationCriterion end

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

struct FactorModelView{M <: AbstractMatrix, L <: AbstractMatrix, V <: AbstractVector, R} <:
       AbstractFactorModel
    "The matrix of factors"
    factors::M
    "The matrix of loadings"
    loadings::L
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

Linear constraints on factor loadings: `R × λᵢ = r` for specified series.

Used with the `:ls` estimation method for sign normalization, zero restrictions,
or identification constraints.

# Constructors

- `LoadingConstraints(series, R, r)` - direct specification
- `LoadingConstraints(matrix)` - each row is `[series, R₁, ..., Rₖ, r]`

See also: [`normalize_loading`](@ref), [`zero_loading`](@ref)

# Examples

Using helper functions (recommended):

```jldoctest
julia> c1 = normalize_loading(1, 1; value=1.0);  # λ₁₁ = 1

julia> c2 = zero_loading(5, 2);  # λ₅₂ = 0

julia> c = vcat(c1, c2);  # combine constraints

julia> c.series
2-element Vector{Int64}:
 1
 5
```

Using matrix format:

```jldoctest
julia> mat = [1.0  1.0  0.0  0.0  1.0];  # series 1: 1*λ₁ + 0*λ₂ + 0*λ₃ = 1

julia> c = LoadingConstraints(mat);

julia> c.series
1-element Vector{Int64}:
 1

julia> c.r
1-element Vector{Float64}:
 1.0
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

"""
    _widen_R(R::Matrix{Float64}, numfactors::Int) -> Matrix{Float64}

Pad columns of `R` with zeros so that it has `numfactors` columns.
"""
function _widen_R(R::Matrix{Float64}, numfactors::Int)
    ncols = size(R, 2)
    ncols == numfactors && return R
    ncols > numfactors &&
        throw(ArgumentError("R has $ncols columns but numfactors=$numfactors"))
    hcat(R, zeros(size(R, 1), numfactors - ncols))
end

function Base.vcat(c1::LoadingConstraints, c2::LoadingConstraints)
    ncols = max(size(c1.R, 2), size(c2.R, 2))
    LoadingConstraints(
        vcat(c1.series, c2.series),
        vcat(_widen_R(c1.R, ncols), _widen_R(c2.R, ncols)),
        vcat(c1.r, c2.r)
    )
end

function Base.vcat(c1::LoadingConstraints, c2::LoadingConstraints, cs::LoadingConstraints...)
    reduce(vcat, cs; init = vcat(c1, c2))
end

"""
    normalize_loading(series, factor; value=1.0)

Constrain loading of `series` on `factor` to equal `value`.

Common use: sign normalization by setting a reference loading to 1.0.

# Examples

```jldoctest
julia> c = normalize_loading(1, 1; value=1.0);

julia> c.R
1×1 Matrix{Float64}:
 1.0

julia> c.r
1-element Vector{Float64}:
 1.0
```

Apply to estimation:

```jldoctest
julia> X = randn(100, 10);

julia> c = normalize_loading(1, 1; value=1.0);

julia> fm = FactorModel(X, 3; constraints=c);

julia> round(loadings(fm)[1, 1], digits=6)
1.0
```

See also: [`zero_loading`](@ref), [`LoadingConstraints`](@ref)
"""
function normalize_loading(series::Int, factor::Int; value::Float64 = 1.0)
    R = zeros(1, factor)
    R[1, factor] = 1.0
    LoadingConstraints([series], R, [value])
end

"""
    zero_loading(series, factor)

Constrain loading of `series` on `factor` to zero.

Use for excluding a variable from loading on a specific factor.

# Examples

```jldoctest
julia> c = zero_loading(5, 2);

julia> c.series
1-element Vector{Int64}:
 5

julia> c.R
1×2 Matrix{Float64}:
 0.0  1.0

julia> c.r
1-element Vector{Float64}:
 0.0
```

Apply to estimation:

```jldoctest
julia> X = randn(100, 10);

julia> c = zero_loading(3, 2);

julia> fm = FactorModel(X, 3; constraints=c);

julia> abs(loadings(fm)[3, 2]) < 1e-10
true
```

See also: [`normalize_loading`](@ref), [`LoadingConstraints`](@ref)
"""
function zero_loading(series::Int, factor::Int)
    normalize_loading(series, factor; value = 0.0)
end

"""
    fix_loading(series, values)

Fix the entire loading vector of `series` to `values`.

The number of factors is inferred from `length(values)`.
Creates one constraint per entry: λ_{series,k} = values[k]
for k = 1, …, length(values).

# Examples

```jldoctest
julia> c = fix_loading(4, [0, 0, 1, 0, 0, 0]);

julia> length(c.series)
6

julia> c.r
6-element Vector{Float64}:
 0.0
 0.0
 1.0
 0.0
 0.0
 0.0
```

See also: [`normalize_loading`](@ref), [`zero_loading`](@ref), [`identity_loading`](@ref)
"""
function fix_loading(series::Int, values::AbstractVector)
    numfactors = length(values)
    R = Matrix{Float64}(I, numfactors, numfactors)
    LoadingConstraints(fill(series, numfactors), R, Float64.(values))
end

"""
    identity_loading(named_series)

Apply the named-factor (identity) normalization: for each k = 1, …, r,
fix the loading of `named_series[k]` to the k-th standard basis vector eₖ,
where r = length(named_series).

This imposes r² restrictions (r unit entries on the diagonal, r²-r zeros off
the diagonal), exactly resolving the rotational indeterminacy of the factor model.

# Examples

```jldoctest
julia> c = identity_loading([1, 2, 3]);

julia> length(c.series)
9

julia> c.r
9-element Vector{Float64}:
 1.0
 0.0
 0.0
 0.0
 1.0
 0.0
 0.0
 0.0
 1.0
```

See also: [`fix_loading`](@ref), [`normalize_loading`](@ref)
"""
function identity_loading(named_series::AbstractVector{<:Integer})
    numfactors = length(named_series)
    constraints = fix_loading(named_series[1], Float64.(I(numfactors)[:, 1]))
    for k in 2:numfactors
        constraints = vcat(constraints,
            fix_loading(named_series[k], Float64.(I(numfactors)[:, k])))
    end
    constraints
end

function FactorModel(Z::AbstractMatrix{G}; kwargs...) where {G}
    FactorModel(Z, size(Z, 2); kwargs...)
end

"""
    FactorModel(Z, numfactors; demean=true, scale=false, method=:auto, ...)

Estimate a static factor model ``X = FΛ' + ε`` where `F` is T×r factors and `Λ` is n×r loadings.

The method is auto-selected based on data and constraints:
- Complete data, no constraints → PCA
- Missing values (NaN) → EM algorithm
- Constraints provided → Iterative Least Squares

# Arguments
- `Z::AbstractMatrix`: T×n data matrix (rows=observations, columns=variables)
- `numfactors::Int`: number of factors r to extract
- `demean::Bool=true`: center columns by their means
- `scale::Bool=false`: standardize columns to unit variance
- `method::Symbol=:auto`: force `:pca`, `:em`, or `:ls` (default: auto-select)
- `constraints::LoadingConstraints=nothing`: linear constraints on loadings (requires `:ls`)
- `maxiter::Int=1000`: max iterations for EM/LS algorithms
- `tol::Float64=1e-8`: convergence tolerance
- `ic::Union{Nothing,Type{<:AbstractInformationCriterion}}=nothing`: if set (e.g., `IC2`),
  use IC-adaptive factor selection inside the EM loop. `numfactors` acts as `kmax` and the
  IC selects `r` at each iteration. Only active when the EM algorithm runs (i.e., data has
  missing values). Cannot be combined with constraints or non-EM methods.

# Examples

Basic usage with PCA (complete data):

```jldoctest
julia> X = randn(100, 10);

julia> fm = FactorModel(X, 3);

julia> size(factors(fm))
(100, 3)

julia> size(loadings(fm))
(10, 3)

julia> estimationmethod(fm)
PCA()
```

With standardization:

```jldoctest
julia> X = randn(100, 10);

julia> fm = FactorModel(X, 3; scale=true);

julia> numfactors(fm)
3
```

Missing data triggers EM automatically:

```jldoctest
julia> X = randn(50, 8); X[1:5, 1] .= NaN;

julia> fm = FactorModel(X, 2);

julia> estimationmethod(fm)
EM()
```

With loading constraints:

```jldoctest
julia> X = randn(100, 10);

julia> c = normalize_loading(1, 1; value=1.0);

julia> fm = FactorModel(X, 3; constraints=c);

julia> estimationmethod(fm)
LeastSquares()

julia> round(loadings(fm)[1, 1], digits=6)
1.0
```

See also: [`factors`](@ref), [`loadings`](@ref), [`estimationmethod`](@ref), [`LoadingConstraints`](@ref)
"""
function FactorModel(Z::AbstractMatrix{G}, numfactors;
        demean::Bool = true, scale::Bool = false, corrected::Bool = false,
        init = nanmean, maxiter::Int = 1000, tol::Float64 = 1e-8,
        constraints::Union{Nothing, LoadingConstraints} = nothing,
        method::Symbol = :auto, nt_min::Int = 10,
        orthonormalize::Bool = true,
        ic::Union{Nothing, Type{<:AbstractInformationCriterion}} = nothing) where {G}
    T, n = size(Z)
    T == 0 && throw(ArgumentError("Input matrix must not be empty (got size $(size(Z)))"))
    numfactors < 0 &&
        throw(ArgumentError("numfactors must be non-negative (got $numfactors)"))
    numfactors > n &&
        throw(ArgumentError("numfactors ($numfactors) must not exceed number of columns ($n)"))

    # Auto-detect missing values
    has_missing = any(isnan, Z)
    has_constraints = constraints !== nothing

    # Validate ic kwarg
    if ic !== nothing
        has_constraints &&
            throw(ArgumentError("ic cannot be used with constraints. Use ic=nothing or remove constraints"))
    end

    # Validate constraints
    if has_constraints
        max_series = maximum(constraints.series)
        max_series > n &&
            throw(ArgumentError("Constraint references series $max_series but data has only $n columns"))
        if size(constraints.R, 2) < numfactors
            constraints = LoadingConstraints(constraints.series,
                _widen_R(constraints.R, numfactors), constraints.r)
        elseif size(constraints.R, 2) > numfactors
            throw(ArgumentError(
                "Constraint R matrix has $(size(constraints.R, 2)) columns but numfactors=$numfactors"))
        end
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

    # Validate ic with explicit method choice
    if ic !== nothing
        actual_method == :pca &&
            throw(ArgumentError("ic cannot be used with method=:pca. IC-adaptive selection requires the EM algorithm"))
        actual_method == :ls &&
            throw(ArgumentError("ic cannot be used with method=:ls. IC-adaptive selection requires the EM algorithm"))
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
            init = init, maxiter = maxiter, tol = tol, ic = ic)
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
    (X, μ, σₓ) = _standardize(Z; demean = demean, scale = scale, corrected = corrected)

    (F, Λ, λ) = _pca(X, numfactors)
    ε = X .- F * Λ'

    # Compute estimation statistics
    tss = sum(abs2, X)
    ssr = sum(abs2, ε)
    nobs = T * n
    r2vec = _compute_r2vec(X, ε)

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
## Internal Helper Functions
## ------------------------------------------------------------

# Standardize data: center by mean and optionally scale by std
function _standardize(Z::AbstractMatrix;
        demean::Bool = true, scale::Bool = false, corrected::Bool = false,
        nan_aware::Bool = false)
    n = size(Z, 2)
    if nan_aware
        μ = demean ? nanmean(Z; dims = 1) : zeros(1, n)
        σₓ = scale ? nanstd(Z; dims = 1, corrected = corrected) : ones(1, n)
    else
        μ = demean ? mean(Z; dims = 1) : zeros(1, n)
        σₓ = scale ? std(Z; dims = 1, corrected = corrected) : ones(1, n)
    end
    X = (Z .- μ) ./ σₓ
    return (X, μ, σₓ)
end

# Compute R² for each series (no NaN in standardized data)
function _compute_r2vec(X::AbstractMatrix, ε::AbstractMatrix)
    n = size(X, 2)
    r2vec = Vector{Float64}(undef, n)
    for i in 1:n
        tss_i = sum(abs2, @view X[:, i])
        ssr_i = sum(abs2, @view ε[:, i])
        r2vec[i] = 1.0 - ssr_i / tss_i
    end
    return r2vec
end

# Compute TSS and nobs, skipping NaN values
function _sum_squares_skipnan(X::AbstractMatrix)
    tss = zero(eltype(X))
    nobs = 0
    for val in X
        if !isnan(val)
            tss += val^2
            nobs += 1
        end
    end
    return (tss, nobs)
end

# Compute SSR skipping NaN values in X
function _ssr_skipnan(X::AbstractMatrix, F::AbstractMatrix, Λ::AbstractMatrix)
    T, n = size(X)
    ssr = zero(eltype(X))
    for i in 1:n, t in 1:T

        val = X[t, i]
        if !isnan(val)
            pred = dot(@view(F[t, :]), @view(Λ[i, :]))
            ssr += (val - pred)^2
        end
    end
    return ssr
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
        init = nanmean, maxiter::Int = 1000, tol::Float64 = 1e-8,
        ic::Union{Nothing, Type{<:AbstractInformationCriterion}} = nothing)
    T, n = size(Z)

    # 1. Identify missing values
    missing_mask = isnan.(Z)
    has_missing = any(missing_mask)
    missing_indices = findall(missing_mask)

    # IC-adaptive mode: numfactors acts as kmax, IC selects r per iteration
    # Only active when ic !== nothing AND there are missing values (EM loop runs)
    ic_adaptive = ic !== nothing && has_missing
    kmax = numfactors  # PCA always uses kmax columns

    # 2. Fill missing values with unconditional column means in raw data
    #    (following McCracken & Ng MATLAB code: fill NaN first, then standardize)
    Z_filled = copy(Z)
    if has_missing
        for j in 1:n
            col_mask = @view missing_mask[:, j]
            if any(col_mask)
                col_data = @view Z[.!col_mask, j]
                fill_val = isempty(col_data) ? zero(eltype(Z)) : init(col_data)
                Z_filled[col_mask, j] .= fill_val
            end
        end
    end

    # 3. Standardize the filled data
    (X, μ,
        σₓ) = _standardize(Z_filled; demean = demean, scale = scale,
        corrected = corrected)

    # Track IC-selected r across iterations (for final output)
    r_selected = kmax

    if has_missing
        # 4. Initial PCA to get predicted values
        (F, Λ, λ) = _pca(X, kmax)

        # For IC-adaptive: select r and use only r factors for chat
        if ic_adaptive
            r_selected = _select_numfactors_ic(ic, X, F, Λ, kmax)
            chat = @views F[:, 1:r_selected] * Λ[:, 1:r_selected]'
        else
            chat = F * Λ'
        end
        chat0 = copy(chat)

        # 5. EM iteration (following Stock-Watson / McCracken-Ng algorithm):
        #    - Update missing values in raw data using predicted values
        #    - Re-standardize the updated data
        #    - Re-estimate factors
        #    - Repeat until convergence
        converged = false
        err = Inf

        for iter in 1:maxiter
            # E-step: Update missing values in raw data
            for idx in missing_indices
                i, j = idx[1], idx[2]
                # Undo standardization: x_raw = chat * σ + μ
                Z_filled[idx] = chat[idx] * σₓ[1, j] + μ[1, j]
            end

            # M-step: Re-standardize and re-estimate
            (X,
                μ,
                σₓ) = _standardize(Z_filled; demean = demean, scale = scale,
                corrected = corrected)
            (F, Λ, λ) = _pca(X, kmax)

            # For IC-adaptive: select r and use only r factors for chat
            if ic_adaptive
                r_selected = _select_numfactors_ic(ic, X, F, Λ, kmax)
                chat = @views F[:, 1:r_selected] * Λ[:, 1:r_selected]'
            else
                chat = F * Λ'
            end

            # Convergence check: relative change in predicted values
            diff_vec = chat .- chat0
            err = sum(abs2, diff_vec) / max(sum(abs2, chat0), eps())

            chat0 .= chat

            if err < tol
                converged = true
                break
            end
        end

        if !converged
            @warn "EM algorithm did not converge after $maxiter iterations (final change: $(err))"
        end
    end

    # 6. Final PCA on converged standardized data
    (F, Λ, λ) = _pca(X, kmax)

    # For IC-adaptive: do a final IC selection and trim to selected r
    if ic_adaptive
        r_selected = _select_numfactors_ic(ic, X, F, Λ, kmax)
        F = F[:, 1:r_selected]
        Λ = Λ[:, 1:r_selected]
        λ = λ[1:r_selected]
    end

    ε = X .- F * Λ'

    # Compute estimation statistics (on standardized, imputed data)
    tss = sum(abs2, X)
    ssr = sum(abs2, ε)
    nobs = count(!isnan, Z)  # non-missing observations in original data
    r2vec = _compute_r2vec(X, ε)

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

    # Standardize using NaN-aware statistics (MATLAB uses population std)
    (X,
        μ,
        σₓ) = _standardize(Z; demean = demean, scale = scale, corrected = false,
        nan_aware = true)

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
    (tss, nobs) = _sum_squares_skipnan(X)

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
        ssr = _ssr_skipnan(X, F, Λ)

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

Return the number of factors r in the model.

# Examples

```jldoctest
julia> X = randn(100, 10);

julia> fm = FactorModel(X, 5);

julia> numfactors(fm)
5
```
"""
numfactors(fm::AbstractFactorModel) = size(loadings(fm), 2)

"""
    loadings(fm::FactorModel; original_units::Bool=false)

Return the n×r loadings matrix Λ where each row contains a variable's loadings on the r factors.

With default normalization, loadings are orthonormal: `Λ'Λ ≈ I`.

# Arguments
- `original_units::Bool=false`: if `true` and `scale=true` was used, return loadings
  in original data units (Λ .* σ)

# Examples

```jldoctest
julia> X = randn(100, 10);

julia> fm = FactorModel(X, 3);

julia> Λ = loadings(fm);

julia> size(Λ)
(10, 3)
```

Loadings are orthonormal by default:

```jldoctest
julia> X = randn(100, 10);

julia> fm = FactorModel(X, 3);

julia> Λ = loadings(fm);

julia> round.(Λ' * Λ, digits=10) ≈ I(3)
true
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

Return the T×r matrix F of estimated factors where each column is a latent factor time series.

# Examples

```jldoctest
julia> X = randn(100, 10);

julia> fm = FactorModel(X, 3);

julia> F = factors(fm);

julia> size(F)
(100, 3)
```
"""
factors(fm::AbstractFactorModel) = fm.factors

LinearAlgebra.eigvals(fm::AbstractFactorModel) = fm.eigenvalues

"""
    sdev(fm::AbstractFactorModel)

Return the standard deviations of the factors, i.e., `sqrt.(eigvals(fm))`.

These are the square roots of the eigenvalues of `cov(X, corrected=false)`.

# Examples

```jldoctest
julia> X = randn(100, 10);

julia> fm = FactorModel(X, 3);

julia> sd = sdev(fm);

julia> length(sd)
3

julia> sd ≈ sqrt.(eigvals(fm))
true
```
"""
function sdev(fm::AbstractFactorModel)
    λ = eigvals(fm)
    sqrt.(λ)
end

"""
    explained_variance(fm::FactorModel)

Return the proportion of total variance explained by each factor.

The sum gives the fraction of variance captured by all r factors (typically < 1).

# Examples

```jldoctest
julia> X = randn(100, 10);

julia> fm = FactorModel(X, 3);

julia> ev = explained_variance(fm);

julia> length(ev)
3

julia> all(ev .>= 0)
true

julia> sum(ev) <= 1.0
true
```
"""
function explained_variance(fm::FactorModel)
    T, _ = size(fm)
    λ = eigvals(fm)
    # Total variance = sum of all eigenvalues of cov(X) = sum(abs2, X̄) / T
    total_var = sum(abs2, fm.X̄) / T
    λ ./ total_var
end

"""
    residuals(fm::AbstractFactorModel)

Compute the residuals of the factor model, defined as `X̄ - F * Λ'` where `X̄` is the
demeaned (and possibly scaled) data matrix, `F` is the matrix of factors, and `Λ` is the
matrix of loadings. The result is stored in-place in `fm.residuals` and returned.
"""
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

Return the estimation method used: `PCA()`, `EM()`, or `LeastSquares()`.

Useful for verifying which algorithm was auto-selected, or for dispatch.

# Examples

```jldoctest
julia> X = randn(100, 10);

julia> fm = FactorModel(X, 3);

julia> estimationmethod(fm)
PCA()

julia> fm_ls = FactorModel(X, 3; method=:ls);

julia> estimationmethod(fm_ls)
LeastSquares()
```

Dispatch on estimation method:

```jldoctest
julia> X = randn(50, 8);

julia> describe_method(::FactorModel{PCA}) = "PCA";

julia> describe_method(::FactorModel{EM}) = "EM";

julia> fm = FactorModel(X, 2);

julia> describe_method(fm)
"PCA"
```
"""
estimationmethod(::FactorModel{E}) where {E} = E()

## Output

# Helper functions for method names
_method_name(::Type{PCA}) = "PCA"
_method_name(::Type{EM}) = "EM"
_method_name(::Type{LeastSquares}) = "LS"

function Base.show(io::IO, fm::FactorModel{E}) where {E}
    T, n = size(fm)
    r = numfactors(fm)
    print(io, "FactorModel{$(_method_name(E))} with $r factor$(r == 1 ? "" : "s") ($T × $n)")
end

function Base.show(io::IO, fmv::FactorModelView)
    T, n = size(fmv.X̄)
    r = numfactors(fmv)
    print(io, "FactorModelView with $r factor$(r == 1 ? "" : "s") ($T × $n)")
end

"""
    describe(fm::FactorModel)
    describe(io::IO, fm::FactorModel)

Print a detailed summary of the factor model, including:
- Model dimensions (T × n)
- Number of factors
- Estimation method
- Factor importance table with standard deviations, proportion of variance,
  and cumulative proportion

# Example
```julia
fm = FactorModel(X, 3)
describe(fm)
```
"""
describe(fm::FactorModel) = describe(stdout, fm)

function describe(io::IO, fm::FactorModel{E}) where {E}
    T, n = size(fm)
    k = numfactors(fm)
    ev = explained_variance(fm)
    cumev = cumsum(ev)
    sd = sdev(fm)

    # Build the data matrix: rows = factors, columns = Std. Dev, Prop. Variance (%), Cumulative (%)
    mat = hcat(sd, 100.0 .* ev, 100.0 .* cumev)

    # Row labels
    row_labels = ["Factor $i" for i in 1:k]

    # Highlighter: bold the cumulative column when it reaches ≥ 90%
    hl = TextHighlighter(
        (data, i, j) -> j == 3 && data[i, 3] >= 90.0,
        crayon"bold"
    )

    # Format: 4 decimals for std dev, 2 decimals + % for variance columns
    fmt = (v, i, j) -> j == 1 ? @sprintf("%.4f", v) : @sprintf("%.2f%%", v)

    pretty_table(io, mat;
        title = "Static Factor Model",
        subtitle = "Dimensions: $T × $n  ⋅  Factors: $k  ⋅  Method: $(_method_name(E))",
        row_labels = row_labels,
        column_labels = ["Std. Dev.", "Prop. Variance", "Cumul. Variance"],
        formatters = [fmt],
        highlighters = [hl],
        alignment = [:r, :r, :r]
    )
end

## ------------------------------------------------------------
## Information Criteria
## ------------------------------------------------------------

"IC1 criterion from Bai and Ng (2002). Penalty: `(N+T)/(NT) * log(NT/(N+T))`"
struct IC1 <: AbstractInformationCriterion end

"IC2 criterion from Bai and Ng (2002). Penalty: `(N+T)/(NT) * log(min(N,T))`"
struct IC2 <: AbstractInformationCriterion end

"IC3 criterion from Bai and Ng (2002). Penalty: `log(min(N,T)) / min(N,T)`"
struct IC3 <: AbstractInformationCriterion end

"PCp1 criterion (panel Cp variant). Same penalty as IC1."
struct PCp1 <: AbstractInformationCriterion end

"PCp2 criterion (panel Cp variant). Same penalty as IC2."
struct PCp2 <: AbstractInformationCriterion end

"PCp3 criterion (panel Cp variant). Same penalty as IC3."
struct PCp3 <: AbstractInformationCriterion end

"AIC1 criterion. Penalty: `2/T`"
struct AIC1 <: AbstractInformationCriterion end

"AIC2 criterion. Penalty: `2/N`"
struct AIC2 <: AbstractInformationCriterion end

"AIC3 criterion. Penalty: `2*(N+T-k)/(NT)`"
struct AIC3 <: AbstractInformationCriterion end

"BIC1 criterion. Penalty: `log(T)/T`"
struct BIC1 <: AbstractInformationCriterion end

"BIC2 criterion. Penalty: `log(N)/N`"
struct BIC2 <: AbstractInformationCriterion end

"BIC3 criterion. Penalty: `(N+T-k)*log(NT)/(NT)`"
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

"""
    _select_numfactors_ic(ICType, X, F, Λ, kmax) -> Int

Internal helper: select the number of factors via information criterion `ICType`
using pre-computed PCA results (F, Λ from `_pca(X, kmax)`).

Returns the optimal number of factors (at least 1).
"""
function _select_numfactors_ic(
        ::Type{M}, X::AbstractMatrix, F::AbstractMatrix,
        Λ::AbstractMatrix, kmax::Int) where {M <: AbstractInformationCriterion}
    T, n = size(X)
    nobs = T * n

    # V(0): average squared value of X (no factors)
    V0 = sum(abs2, X) / nobs

    # V(k) for k = 1..kmax: average squared residual using first k factors
    Vk = Vector{Float64}(undef, kmax)
    for k in 1:kmax
        Fk = @view F[:, 1:k]
        Λk = @view Λ[:, 1:k]
        ssr_k = sum(abs2, X .- Fk * Λk')
        Vk[k] = ssr_k / nobs
    end

    VV = [V0; Vk]  # length kmax+1, indices correspond to k=0..kmax

    # Apply transform (log for IC1/IC2/IC3, identity for PCp/AIC/BIC)
    Vₖ = transform_V.(M, VV)

    # Variance factor (1.0 for IC1/IC2/IC3, V(kmax) for others)
    σ̂² = M <: Union{IC1, IC2, IC3} ? 1.0 : Vk[kmax]

    # Penalty
    gₜₙ = map(k -> k * penalty(M, T, n, k), 0:kmax)

    # Criterion values for k = 0..kmax
    crit = Vₖ .+ σ̂² .* gₜₙ

    # Select k that minimizes criterion (at least 1)
    _, idx = findmin(crit)
    r = idx - 1  # 0-indexed: idx=1 corresponds to k=0
    return max(r, 1)
end

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

"""
    informationcriteria(criteria::Tuple, fm::FactorModel, kmax::Int)

Compute multiple information criteria simultaneously for model selection.

Returns a tuple of `InformationCriterion` objects, one per criterion type.

# Examples

```jldoctest
julia> X = randn(100, 20);

julia> fm = FactorModel(X, 8);

julia> ics = informationcriteria((IC1, IC2, BIC1), fm, 8);

julia> length(ics)
3

julia> numfactors(ics[1]) >= 0
true
```

Compare multiple criteria:

```jldoctest
julia> X = randn(100, 20);

julia> fm = FactorModel(X, 5);

julia> ics = informationcriteria((IC1, IC2), fm, 5);

julia> results = findmin(ics);

julia> length(results)
2
```

See also: [`IC1`](@ref), [`IC2`](@ref), [`IC3`](@ref), [`criterion`](@ref), [`numfactors`](@ref)
"""
function informationcriteria(criteria::Tuple, fm, kₘₐₓ)
    for c in criteria
        isa(c(), AbstractInformationCriterion) ||
            throw(ArgumentError("$(c) is not an AbstractInformationCriterion"))
    end
    map(c -> c(fm, kₘₐₓ), criteria)
end

for criterion in
    (:BIC1, :BIC2, :BIC3, :AIC1, :AIC2, :AIC3, :IC1, :IC2, :IC3, :PCp1, :PCp2, :PCp3)
    @eval begin
        function ($criterion)(fm, kₘₐₓ::Int64)
            informationcriterion($criterion, fm, kₘₐₓ)
        end
        Base.string(::($criterion)) = $(string(criterion))
        function ($criterion)(X::Matrix, kₘₐₓ::Int64; kwargs...)
            informationcriterion($criterion, FactorModel(X, kₘₐₓ; kwargs...), kₘₐₓ)
        end
    end
end

function Base.findmin(ic::InformationCriterion)
    val, idx = findmin(ic.crit)
    k = ic.rnge[idx]
    NamedTuple{(Symbol(string(ic.criterion)), :r)}((val, k))
end

function Base.findmin(ic::Tuple{Vararg{InformationCriterion, N}}) where {N}
    fmin = map(x -> begin
            val, idx = findmin(x.crit)
            k = x.rnge[idx]
            (val, k)
        end, ic)
    nm = map(x->(Symbol(x.criterion), :r), ic)
    TT = map(x->eltype(x.crit), ic)
    map((nm, x, T) -> NamedTuple{nm}(x), nm, fmin, TT)
end

numfactors(ic::InformationCriterion) = findmin(ic).r
Base.string(ic::InformationCriterion{T, F}) where {T, F} = string(T)

"""
    criterion(ic::InformationCriterion)

Return the vector of criterion values for r = 0, 1, ..., kmax factors.

The optimal r minimizes this vector (use `numfactors(ic)` or `findmin(ic)`).

# Examples

```jldoctest
julia> X = randn(100, 10);

julia> fm = FactorModel(X, 5);

julia> ic = IC1(fm, 5);

julia> vals = criterion(ic);

julia> length(vals)  # values for r = 0, 1, 2, 3, 4, 5
6

julia> r_opt = numfactors(ic);

julia> r_opt >= 0 && r_opt <= 5
true
```

See also: [`numfactors`](@ref), `findmin`
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

## ------------------------------------------------------------
## R² Display Types and Functions
## ------------------------------------------------------------

"""
    TotalR2

Result type for total R² (all factors) per variable.

Implements the Tables.jl interface so you can convert to a DataFrame:

```julia
using DataFrames
DataFrame(total_r2(fm))
```
"""
struct TotalR2
    varnames::Vector{String}
    r2::Vector{Float64}
    show_all::Bool
end

"""
    ByFactorR2

Result type for per-factor R² per variable.

Implements the Tables.jl interface so you can convert to a DataFrame:

```julia
using DataFrames
DataFrame(byfactor_r2(fm))
```
"""
struct ByFactorR2
    varnames::Vector{String}
    r2mat::Matrix{Float64}   # n × r
    factornames::Vector{String}
    show_all::Bool
end

# --- Tables.jl interface for TotalR2 ---

Tables.istable(::Type{TotalR2}) = true
Tables.columnaccess(::Type{TotalR2}) = true

function Tables.columns(t::TotalR2)
    return t
end

function Tables.columnnames(t::TotalR2)
    return (:Variable, :R2)
end

function Tables.getcolumn(t::TotalR2, nm::Symbol)
    if nm === :Variable
        return t.varnames
    elseif nm === :R2
        return t.r2
    else
        throw(ArgumentError("TotalR2 has no column $nm"))
    end
end

Tables.getcolumn(t::TotalR2, i::Int) = Tables.getcolumn(t, Tables.columnnames(t)[i])

function Tables.schema(t::TotalR2)
    return Tables.Schema((:Variable, :R2), (String, Float64))
end

# --- Tables.jl interface for ByFactorR2 ---

Tables.istable(::Type{ByFactorR2}) = true
Tables.columnaccess(::Type{ByFactorR2}) = true

function Tables.columns(t::ByFactorR2)
    return t
end

function Tables.columnnames(t::ByFactorR2)
    return (Symbol("Variable"), Symbol.(t.factornames)...)
end

function Tables.getcolumn(t::ByFactorR2, nm::Symbol)
    if nm === :Variable
        return t.varnames
    else
        idx = findfirst(==(String(nm)), t.factornames)
        if idx !== nothing
            return t.r2mat[:, idx]
        else
            throw(ArgumentError("ByFactorR2 has no column $nm"))
        end
    end
end

Tables.getcolumn(t::ByFactorR2, i::Int) = Tables.getcolumn(t, Tables.columnnames(t)[i])

function Tables.schema(t::ByFactorR2)
    names = Tables.columnnames(t)
    types = (String, (Float64 for _ in t.factornames)...)
    return Tables.Schema(names, types)
end

# --- User-facing functions ---

"""
    total_r2(fm::FactorModel; varnames=nothing, show_all=false)

Return a `TotalR2` object showing per-variable R² (all factors combined).

# Arguments
- `varnames`: optional vector of variable names (default: `["V1", "V2", ...]`)
- `show_all`: if `true`, display all variables; otherwise show top/bottom 10%

# Examples

```julia
fm = FactorModel(randn(100, 20), 3; scale=true)
total_r2(fm)                    # shows top/bottom 10%
total_r2(fm; show_all=true)     # shows all variables

using DataFrames
DataFrame(total_r2(fm))         # convert to DataFrame
```
"""
function total_r2(fm::FactorModel;
        varnames::Union{Nothing, Vector{String}} = nothing,
        show_all::Bool = false)
    r2vec = r2(fm)
    n = length(r2vec)
    if varnames === nothing
        varnames = ["V$i" for i in 1:n]
    end
    length(varnames) != n &&
        throw(ArgumentError("varnames has length $(length(varnames)) but model has $n variables"))
    TotalR2(varnames, r2vec, show_all)
end

"""
    byfactor_r2(fm::FactorModel; varnames=nothing, show_all=false)

Return a `ByFactorR2` object showing per-variable, per-factor R².

For variable `i` and factor `j`:
    R²ᵢⱼ = 1 - Σₜ(X̄ₜᵢ - Fₜⱼ·Λᵢⱼ)² / Σₜ X̄ₜᵢ²

# Arguments
- `varnames`: optional vector of variable names (default: `["V1", "V2", ...]`)
- `show_all`: if `true`, display all variables; otherwise show top/bottom 10%

# Examples

```julia
fm = FactorModel(randn(100, 20), 3; scale=true)
byfactor_r2(fm)                  # shows per-factor R²

using DataFrames
DataFrame(byfactor_r2(fm))       # convert to DataFrame
```
"""
function byfactor_r2(fm::FactorModel;
        varnames::Union{Nothing, Vector{String}} = nothing,
        show_all::Bool = false)
    F = factors(fm)
    Λ = loadings(fm)
    X_bar = fm.X̄
    n = size(Λ, 1)
    r = size(Λ, 2)

    if varnames === nothing
        varnames = ["V$i" for i in 1:n]
    end
    length(varnames) != n &&
        throw(ArgumentError("varnames has length $(length(varnames)) but model has $n variables"))

    r2mat = Matrix{Float64}(undef, n, r)
    for i in 1:n
        tss_i = sum(abs2, @view X_bar[:, i])
        for j in 1:r
            ssr_ij = sum(t -> (X_bar[t, i] - F[t, j] * Λ[i, j])^2, 1:size(F, 1))
            r2mat[i, j] = 1.0 - ssr_ij / tss_i
        end
    end

    factornames = ["Factor_$j" for j in 1:r]
    ByFactorR2(varnames, r2mat, factornames, show_all)
end

# --- Base.show methods ---

function Base.show(io::IO, t::TotalR2)
    n = length(t.r2)
    printstyled(io, "\nTotal R² (all factors)\n", color = :green)
    @printf io "Number of variables: %d\n\n" n

    # Sort by R² descending
    perm = sortperm(t.r2; rev = true)

    if t.show_all || n <= 20
        rows = perm
        show_message = false
    else
        ntop = max(1, round(Int, 0.1 * n))
        nbot = max(1, round(Int, 0.1 * n))
        rows = vcat(perm[1:ntop], perm[(end - nbot + 1):end])
        show_message = true
    end

    tbl = hcat(t.varnames[rows], t.r2[rows])
    column_labels = ["Variable", "R²"]

    hl_green = TextHighlighter(
        (data, i, j) -> j == 2 && data[i, 2] isa Number && data[i, 2] >= 0.9,
        Crayon(foreground = :green, bold = true))
    hl_red = TextHighlighter(
        (data, i, j) -> j == 2 && data[i, 2] isa Number && data[i, 2] < 0.1,
        Crayon(foreground = :red))
    style = TextTableStyle(first_line_column_label = crayon"yellow bold")

    pretty_table(io, tbl;
        column_labels = [column_labels],
        style = style,
        formatters = [fmt__printf("%8.4f", [2])],
        highlighters = [hl_green, hl_red],
        maximum_number_of_rows = -1)

    if show_message
        println(io, "\nShowing top/bottom 10%. Use show_all=true to see all variables.")
    end
end

function Base.show(io::IO, t::ByFactorR2)
    n = length(t.varnames)
    r = length(t.factornames)
    printstyled(io, "\nR² by Individual Factor\n", color = :green)
    @printf io "Variables: %d, Factors: %d\n\n" n r

    # Rank by total R² (row sum) descending
    rowsums = vec(sum(t.r2mat; dims = 2))
    perm = sortperm(rowsums; rev = true)

    if t.show_all || n <= 20
        rows = perm
        show_message = false
    else
        ntop = max(1, round(Int, 0.1 * n))
        nbot = max(1, round(Int, 0.1 * n))
        rows = vcat(perm[1:ntop], perm[(end - nbot + 1):end])
        show_message = true
    end

    tbl = hcat(t.varnames[rows], t.r2mat[rows, :])
    column_labels = ["Variable"; t.factornames]

    hl_green = TextHighlighter(
        (data, i, j) -> j > 1 && data[i, j] isa Number && data[i, j] >= 0.5,
        Crayon(foreground = :green, bold = true))
    hl_yellow = TextHighlighter(
        (data, i, j) -> j > 1 && data[i, j] isa Number && 0.1 <= data[i, j] < 0.5,
        Crayon(foreground = :yellow))
    style = TextTableStyle(first_line_column_label = crayon"yellow bold")

    fmt_cols = collect(2:(r + 1))
    pretty_table(io, tbl;
        column_labels = [column_labels],
        style = style,
        formatters = [fmt__printf("%8.4f", fmt_cols)],
        highlighters = [hl_green, hl_yellow],
        maximum_number_of_rows = -1)

    if show_message
        println(io,
            "\nShowing top/bottom 10% (ranked by total R²). Use show_all=true to see all variables.")
    end
end

export FactorModel, EstimationStats, describe,
       numfactors, factors, loadings, explained_variance, sdev,
# Estimation method types and accessor
       AbstractEstimationMethod, PCA, EM, LeastSquares, estimationmethod,
# Estimation statistics
       stats, tss, ssr, r2, nobs,
# R² display
       TotalR2, ByFactorR2, total_r2, byfactor_r2,
# Information criteria
       IC1, IC2, IC3, PCp1, PCp2, PCp3,
       AIC1, AIC2, AIC3, BIC1, BIC2, BIC3,
       informationcriteria, criterion,
# Constrained factor estimation
       LoadingConstraints, normalize_loading, zero_loading, fix_loading, identity_loading,
       residuals

end # module"
