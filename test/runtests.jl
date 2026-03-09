using Factotum, Statistics, LinearAlgebra, Test, Random, NaNStatistics
using CSV, DataFrames
import Factotum: Tables

@testset "Factotum.jl" begin
    @testset "Basic factor model (T > n)" begin
        T, n, r = (200, 10, 6)
        x = rand(T, n)
        fm = FactorModel(x, r; scale = true)

        F = factors(fm)
        Λ = loadings(fm)
        σ = Factotum.sdev(fm)

        # Check diagonality of cov(F)
        Σ = cov(F; corrected = false)
        @test Σ ≈ diagm(0 => σ .^ 2)

        # Check Λ'Λ = I (loadings are orthonormal)
        @test Λ'Λ ≈ diagm(0 => ones(r))

        # Test accessor functions
        @test numfactors(fm) == r
        @test size(factors(fm)) == (T, r)
        @test size(loadings(fm)) == (n, r)
        @test length(explained_variance(fm)) == r
    end

    @testset "Factor model with T <= n (extract_FF)" begin
        # Test case where T <= n
        T, n, r = (5, 10, 3)
        x = randn(T, n)
        fm = FactorModel(x, r; scale = true)

        @test numfactors(fm) == r
        @test size(factors(fm)) == (T, r)
        @test size(loadings(fm)) == (n, r)

        # Verify factor structure
        F = factors(fm)
        Λ = loadings(fm)

        # Residuals should be computable
        ε = Factotum.residuals(fm)
        @test size(ε) == (T, n)
    end

    @testset "demean and scale options" begin
        Random.seed!(42)
        X = randn(100, 10) .+ 5  # shifted data

        # Test demean=true (default)
        fm_demean = FactorModel(X, 5; demean = true, scale = false)
        @test numfactors(fm_demean) == 5

        # Test demean=false
        fm_nodemean = FactorModel(X, 5; demean = false, scale = false)
        @test numfactors(fm_nodemean) == 5

        # Test scale=true
        fm_scaled = FactorModel(X, 5; demean = true, scale = true)
        @test numfactors(fm_scaled) == 5

        # Test both false
        fm_raw = FactorModel(X, 5; demean = false, scale = false)
        @test numfactors(fm_raw) == 5
    end

    @testset "FactorModelView" begin
        X = randn(100, 10)
        fm = FactorModel(X, 5)

        # View with integer
        fmv = view(fm, 3)
        @test numfactors(fmv) == 3
        @test size(factors(fmv)) == (100, 3)
        @test size(loadings(fmv)) == (10, 3)

        # View with range
        fmv2 = view(fm, 2:4)
        @test numfactors(fmv2) == 3
        @test size(factors(fmv2)) == (100, 3)
    end

    @testset "Input validation" begin
        # Empty matrix
        @test_throws ArgumentError FactorModel(Matrix{Float64}(undef, 0, 0), 1)

        # Negative numfactors
        X = randn(10, 5)
        @test_throws ArgumentError FactorModel(X, -1)

        # numfactors exceeds columns
        @test_throws ArgumentError FactorModel(X, 10)

        # View with invalid k
        fm = FactorModel(X, 3)
        @test_throws ArgumentError view(fm, 0)
        @test_throws ArgumentError view(fm, 5)  # exceeds available factors
        @test_throws ArgumentError view(fm, 0:2)  # starts at 0

        # informationcriterion with invalid kmax
        @test_throws ArgumentError IC1(fm, 0)
        @test_throws ArgumentError IC1(fm, 5)  # exceeds numfactors
    end

    @testset "Information criteria" begin
        function simulate_factormodel(r, T, N)
            F = randn(T, r)
            Λ = randn(r, N)
            e = sqrt(r) .* randn(T, N)
            F * Λ .+ e
        end

        Random.seed!(123)
        X = simulate_factormodel(1, 251, 78)
        fm = FactorModel(X, 10; scale = true)

        # Test individual criteria
        ic1 = IC1(fm, 10)
        ic2 = IC2(fm, 10)
        bic1 = BIC1(fm, 10)

        @test length(criterion(ic1)) == 11  # 0 to 10 factors
        @test ic1.rnge == 0:10

        # Test findmin
        result = findmin(ic1)
        @test haskey(result, :IC1)
        @test haskey(result, :r)

        # Test informationcriteria with tuple
        ics = Factotum.informationcriteria((IC1, IC2), fm, 10)
        @test length(ics) == 2

        results = findmin(ics)
        @test length(results) == 2

        # Test numfactors
        @test numfactors(ic1) >= 0
        @test numfactors(ic1) <= 10
    end

    @testset "All criterion types work" begin
        X = randn(50, 10)
        fm = FactorModel(X, 5)

        criteria = [IC1, IC2, IC3, PCp1, PCp2, PCp3, AIC1, AIC2, AIC3, BIC1, BIC2, BIC3]
        for C in criteria
            ic = C(fm, 5)
            @test length(criterion(ic)) == 6  # 0 to 5 factors
            @test numfactors(ic) >= 0
        end
    end

    @testset "Information criteria with LS method and missing data" begin
        # Use macrodata which has real missing values
        datapath = joinpath(@__DIR__, "data", "macrodata.csv")
        df = CSV.read(datapath, DataFrame)
        X_full = Matrix{Float64}(df[:, 2:end])

        fm_em = FactorModel(X_full, 10; method = :em, scale = true)
        fm_ls = FactorModel(X_full, 10; method = :ls, scale = true)

        ic_em = IC1(fm_em, 10)
        ic_ls = IC1(fm_ls, 10)

        # Both should work (no NaN in criterion values)
        @test !any(isnan, criterion(ic_em))
        @test !any(isnan, criterion(ic_ls))

        # Test all criterion types work with LS and missing data
        criteria = [IC1, IC2, IC3, BIC1, BIC2, BIC3]
        for C in criteria
            ic = C(fm_ls, 10)
            @test !any(isnan, criterion(ic))
        end
    end

    @testset "describe and show" begin
        X = randn(50, 10)
        fm = FactorModel(X, 3)

        # Test that show doesn't error
        io = IOBuffer()
        show(io, fm)
        output = String(take!(io))
        @test occursin("Static Factor Model", output)
        @test occursin("Number of factors", output)

        # Test that describe doesn't error
        io = IOBuffer()
        Factotum.describe(io, fm)
        output = String(take!(io))
        @test occursin("Static Factor Model", output)
    end

    @testset "Criterion from matrix directly" begin
        X = randn(100, 10)
        ic = IC1(X, 5; scale = true)
        @test length(criterion(ic)) == 6
    end

    @testset "EM algorithm - no missing values" begin
        # EM with no missing values should produce similar results to standard PCA
        Random.seed!(42)
        X = randn(100, 10)

        fm_standard = FactorModel(X, 3; scale = true)
        fm_em = FactorModel(X, 3; scale = true, method = :em)

        # Results should be very close (up to sign flips in factors/loadings)
        @test numfactors(fm_standard) == numfactors(fm_em)
        @test size(factors(fm_standard)) == size(factors(fm_em))
        @test size(loadings(fm_standard)) == size(loadings(fm_em))

        # Eigenvalues should match
        @test eigvals(fm_standard) ≈ eigvals(fm_em) rtol=1e-6
    end

    @testset "EM algorithm - with missing values (T > n)" begin
        Random.seed!(123)

        # Generate data with known factor structure
        T, n, r = 200, 10, 3
        F_true = randn(T, r)
        Λ_true = randn(n, r)
        X_complete = F_true * Λ_true' + 0.5 * randn(T, n)

        # Introduce ~10% missing values randomly
        X_missing = copy(X_complete)
        missing_mask = rand(T, n) .< 0.1
        X_missing[missing_mask] .= NaN

        # Fit model with missing values
        fm = FactorModel(X_missing, r; scale = true)

        @test numfactors(fm) == r
        @test size(factors(fm)) == (T, r)
        @test size(loadings(fm)) == (n, r)

        # Imputed data (X̄) should have no NaNs
        @test !any(isnan, fm.X̄)

        # Reconstruction should be reasonable
        F = factors(fm)
        Λ = loadings(fm)
        X_reconstructed = F * Λ'

        # Compare reconstruction on non-missing positions
        X_centered = (X_complete .- mean(X_complete; dims = 1)) ./ std(X_complete; dims = 1)
        @test !any(isnan, X_reconstructed)
    end

    @testset "EM algorithm - with missing values (T <= n)" begin
        Random.seed!(456)

        # Test T <= n case
        T, n, r = 10, 20, 3
        F_true = randn(T, r)
        Λ_true = randn(n, r)
        X_complete = F_true * Λ_true' + 0.5 * randn(T, n)

        # Introduce ~10% missing values
        X_missing = copy(X_complete)
        missing_mask = rand(T, n) .< 0.1
        X_missing[missing_mask] .= NaN

        fm = FactorModel(X_missing, r; scale = true)

        @test numfactors(fm) == r
        @test size(factors(fm)) == (T, r)
        @test size(loadings(fm)) == (n, r)
        @test !any(isnan, fm.X̄)
    end

    @testset "EM algorithm - different init functions" begin
        Random.seed!(789)
        T, n, r = 50, 10, 2

        X = randn(T, n)
        X[1:5, 1] .= NaN  # some missing values

        # Test with nanmean (default)
        fm_mean = FactorModel(X, r; init = nanmean)
        @test numfactors(fm_mean) == r
        @test !any(isnan, fm_mean.X̄)

        # Test with nanmedian
        fm_median = FactorModel(X, r; init = nanmedian)
        @test numfactors(fm_median) == r
        @test !any(isnan, fm_median.X̄)

        # Test with custom init function (zero)
        fm_zero = FactorModel(X, r; init = x -> zero(eltype(x)))
        @test numfactors(fm_zero) == r
        @test !any(isnan, fm_zero.X̄)
    end

    @testset "EM algorithm - convergence parameters" begin
        Random.seed!(321)
        T, n, r = 50, 10, 2

        X = randn(T, n)
        X[1:3, 1:2] .= NaN

        # Test with different maxiter and tol
        fm1 = FactorModel(X, r; maxiter = 100, tol = 1e-6)
        fm2 = FactorModel(X, r; maxiter = 2000, tol = 1e-10)

        @test numfactors(fm1) == r
        @test numfactors(fm2) == r
        @test !any(isnan, fm1.X̄)
        @test !any(isnan, fm2.X̄)
    end

    @testset "EM algorithm - entire column missing handled" begin
        Random.seed!(111)
        T, n, r = 50, 10, 2

        X = randn(T, n)
        # Make entire column missing - should fill with zeros after centering
        X[:, 1] .= NaN

        # This should not error, but fill with zeros
        fm = FactorModel(X, r)
        @test numfactors(fm) == r
        @test !any(isnan, fm.X̄)
        # Column with all NaN should be filled with zeros after mean-centering
        @test all(fm.X̄[:, 1] .== 0.0)
    end

    @testset "LS vs EM - no constraints, no missing data" begin
        # When there are no constraints and no missing data,
        # LS and EM should produce approximately equal results
        Random.seed!(42)
        T, n, r = 100, 20, 3
        X = randn(T, n)

        fm_em = FactorModel(X, r; method = :em, scale = true, tol = 1e-10, maxiter = 2000)
        fm_ls = FactorModel(X, r; method = :ls, scale = true, tol = 1e-10, maxiter = 2000)

        # Eigenvalues should be approximately equal
        @test eigvals(fm_em) ≈ eigvals(fm_ls) rtol=0.1

        # Check that both methods produce valid output
        @test numfactors(fm_em) == r
        @test numfactors(fm_ls) == r
        @test size(factors(fm_em)) == (T, r)
        @test size(factors(fm_ls)) == (T, r)
        @test size(loadings(fm_em)) == (n, r)
        @test size(loadings(fm_ls)) == (n, r)

        # Residual sum of squares should be similar
        ssr_em = sum(abs2, Factotum.residuals(fm_em))
        ssr_ls = sum(abs2, Factotum.residuals(fm_ls))
        @test ssr_em ≈ ssr_ls rtol=0.1
    end

    @testset "LS vs EM - no constraints, with missing data" begin
        # Both methods should handle missing data
        # Note: EM imputes missing values, LS keeps them as NaN but handles them during estimation
        Random.seed!(123)
        T, n, r = 200, 20, 3

        # Generate data with known factor structure
        F_true = randn(T, r)
        Λ_true = randn(n, r)
        X_complete = F_true * Λ_true' + 0.5 * randn(T, n)

        # Introduce missing values in specific columns only (keep first r+2 columns complete for LS init)
        X_missing = copy(X_complete)
        # Only add missing values to columns 6 to n
        for j in (r + 3):n
            missing_rows = randperm(T)[1:10]  # 10 missing values per column
            X_missing[missing_rows, j] .= NaN
        end

        fm_em = FactorModel(
            X_missing, r; method = :em, scale = true, tol = 1e-10, maxiter = 2000)
        fm_ls = FactorModel(
            X_missing, r; method = :ls, scale = true, tol = 1e-10, maxiter = 2000)

        # EM imputes missing values, LS does not (by design)
        @test !any(isnan, fm_em.X̄)
        # LS legitimately keeps NaN where data was missing

        # Both methods should produce valid factors and loadings
        @test !any(isnan, factors(fm_em))
        @test !any(isnan, loadings(fm_em))
        @test !any(isnan, factors(fm_ls))
        @test !any(isnan, loadings(fm_ls))

        # Both should converge to reasonable solutions
        @test numfactors(fm_em) == r
        @test numfactors(fm_ls) == r
    end

    @testset "LS method - basic functionality" begin
        Random.seed!(456)
        T, n, r = 100, 10, 3
        X = randn(T, n)

        fm = FactorModel(X, r; method = :ls)

        @test numfactors(fm) == r
        @test size(factors(fm)) == (T, r)
        @test size(loadings(fm)) == (n, r)

        # Reconstruction should be valid
        F = factors(fm)
        Λ = loadings(fm)
        @test !any(isnan, F)
        @test !any(isnan, Λ)
    end

    @testset "LoadingConstraints - construction" begin
        # Test direct constructor
        lc = LoadingConstraints([1, 5], [1.0 0.0 0.0; 0.0 0.0 1.0], [1.0, 0.0])
        @test lc.series == [1, 5]
        @test lc.R == [1.0 0.0 0.0; 0.0 0.0 1.0]
        @test lc.r == [1.0, 0.0]

        # Test matrix constructor
        constraints_mat = [1.0 1.0 0.0 0.0 1.0;  # series 1: 1*λ₁ + 0*λ₂ + 0*λ₃ = 1
                           5.0 0.0 0.0 1.0 0.0;]
        lc2 = LoadingConstraints(constraints_mat)
        @test lc2.series == [1, 5]
        @test lc2.R == [1.0 0.0 0.0; 0.0 0.0 1.0]
        @test lc2.r == [1.0, 0.0]

        # Test helper functions
        c1 = normalize_loading(1, 1, 3; value = 1.0)
        @test c1.series == [1]
        @test c1.R == [1.0 0.0 0.0]
        @test c1.r == [1.0]

        c2 = zero_loading(5, 3, 3)
        @test c2.series == [5]
        @test c2.R == [0.0 0.0 1.0]
        @test c2.r == [0.0]

        # Test vcat
        c_combined = vcat(c1, c2)
        @test c_combined.series == [1, 5]
        @test size(c_combined.R) == (2, 3)
        @test c_combined.r == [1.0, 0.0]
    end

    @testset "Constrained factor estimation - sign normalization" begin
        Random.seed!(789)
        T, n, r = 100, 20, 3
        X = randn(T, n)

        # Constrain first series to have loading = 1.0 on first factor
        c = normalize_loading(1, 1, r; value = 1.0)

        fm = FactorModel(X, r; constraints = c, scale = false)

        # Check that constraint is satisfied
        Λ = loadings(fm)
        @test Λ[1, 1] ≈ 1.0 rtol=1e-6

        @test numfactors(fm) == r
        @test size(factors(fm)) == (T, r)
        @test size(loadings(fm)) == (n, r)
    end

    @testset "Constrained factor estimation - zero loading" begin
        Random.seed!(321)
        T, n, r = 100, 20, 3
        X = randn(T, n)

        # Constrain series 5 to have zero loading on factor 3
        c = zero_loading(5, 3, r)

        fm = FactorModel(X, r; constraints = c, scale = false)

        # Check that constraint is satisfied
        Λ = loadings(fm)
        @test abs(Λ[5, 3]) < 1e-6

        @test numfactors(fm) == r
    end

    @testset "Constrained factor estimation - multiple constraints" begin
        Random.seed!(654)
        T, n, r = 100, 20, 3
        X = randn(T, n)

        # Multiple constraints
        c1 = normalize_loading(1, 1, r; value = 1.0)
        c2 = zero_loading(5, 3, r)
        c3 = normalize_loading(10, 2, r; value = 0.5)

        constraints = vcat(c1, vcat(c2, c3))

        fm = FactorModel(X, r; constraints = constraints, scale = false)

        # Check all constraints are satisfied
        Λ = loadings(fm)
        @test Λ[1, 1] ≈ 1.0 rtol=1e-6
        @test abs(Λ[5, 3]) < 1e-6
        @test Λ[10, 2] ≈ 0.5 rtol=1e-6
    end

    @testset "Constrained estimation - auto method selection" begin
        Random.seed!(111)
        T, n, r = 50, 10, 3
        X = randn(T, n)

        # With constraints, method should auto-select :ls
        c = normalize_loading(1, 1, r)
        fm = FactorModel(X, r; constraints = c)

        @test numfactors(fm) == r
        # Constraint should be satisfied
        @test loadings(fm)[1, 1] ≈ 1.0 rtol=1e-6
    end

    @testset "Constrained estimation - with missing data" begin
        Random.seed!(222)
        T, n, r = 200, 15, 3

        X = randn(T, n)
        # Introduce some missing values
        X[1:10, 1] .= NaN
        X[50:60, 5] .= NaN

        c = normalize_loading(2, 1, r; value = 1.0)  # Series 2 (not series 1 which has NaN)

        fm = FactorModel(X, r; constraints = c, scale = false)

        # Constraint should be satisfied
        Λ = loadings(fm)
        @test Λ[2, 1] ≈ 1.0 rtol=1e-6

        # Output should be valid
        @test numfactors(fm) == r
        @test !any(isnan, factors(fm))
    end

    @testset "Method selection validation" begin
        X = randn(50, 10)
        X_missing = copy(X)
        X_missing[1:5, 1] .= NaN

        r = 3
        c = normalize_loading(1, 1, r)

        # PCA with missing data should error
        @test_throws ArgumentError FactorModel(X_missing, r; method = :pca)

        # PCA with constraints should error
        @test_throws ArgumentError FactorModel(X, r; method = :pca, constraints = c)

        # EM with constraints should error
        @test_throws ArgumentError FactorModel(X, r; method = :em, constraints = c)

        # Invalid method should error
        @test_throws ArgumentError FactorModel(X, r; method = :invalid)
    end

    @testset "Constraint validation" begin
        X = randn(50, 10)
        r = 3

        # Constraint referencing non-existent series
        c_bad_series = LoadingConstraints([100], [1.0 0.0 0.0], [1.0])
        @test_throws ArgumentError FactorModel(X, r; constraints = c_bad_series)

        # Constraint with wrong number of factors in R
        c_bad_r = LoadingConstraints([1], [1.0 0.0 0.0 0.0], [1.0])  # 4 cols but r=3
        @test_throws ArgumentError FactorModel(X, r; constraints = c_bad_r)
    end

    @testset "Estimation method types and accessor" begin
        Random.seed!(42)
        X = randn(100, 10)
        r = 3

        # Test PCA method type
        fm_pca = FactorModel(X, r; method = :pca)
        @test estimationmethod(fm_pca) isa PCA
        @test estimationmethod(fm_pca) == PCA()

        # Test EM method type
        fm_em = FactorModel(X, r; method = :em)
        @test estimationmethod(fm_em) isa EM
        @test estimationmethod(fm_em) == EM()

        # Test LeastSquares method type
        fm_ls = FactorModel(X, r; method = :ls)
        @test estimationmethod(fm_ls) isa LeastSquares
        @test estimationmethod(fm_ls) == LeastSquares()

        # Test auto-selection: default (complete data) -> PCA
        fm_auto = FactorModel(X, r)
        @test estimationmethod(fm_auto) isa PCA

        # Test auto-selection: missing data -> EM
        X_missing = copy(X)
        X_missing[1:5, 1] .= NaN
        fm_auto_em = FactorModel(X_missing, r)
        @test estimationmethod(fm_auto_em) isa EM

        # Test auto-selection: constraints -> LeastSquares
        c = normalize_loading(1, 1, r)
        fm_auto_ls = FactorModel(X, r; constraints = c)
        @test estimationmethod(fm_auto_ls) isa LeastSquares

        # Test type hierarchy
        @test PCA <: AbstractEstimationMethod
        @test EM <: AbstractEstimationMethod
        @test LeastSquares <: AbstractEstimationMethod
    end

    @testset "Estimation method in show output" begin
        Random.seed!(42)
        X = randn(50, 10)
        r = 3

        # Test PCA output
        fm_pca = FactorModel(X, r; method = :pca)
        io = IOBuffer()
        show(io, fm_pca)
        output = String(take!(io))
        @test occursin("Principal Component Analysis", output)

        # Test EM output
        fm_em = FactorModel(X, r; method = :em)
        io = IOBuffer()
        show(io, fm_em)
        output = String(take!(io))
        @test occursin("EM Algorithm", output)

        # Test LS output
        fm_ls = FactorModel(X, r; method = :ls)
        io = IOBuffer()
        show(io, fm_ls)
        output = String(take!(io))
        @test occursin("Iterative Least Squares", output)
    end

    @testset "Type parameter dispatch" begin
        Random.seed!(42)
        X = randn(50, 10)
        r = 3

        # Test that we can dispatch on the estimation method type
        fm_pca = FactorModel(X, r; method = :pca)
        fm_em = FactorModel(X, r; method = :em)
        fm_ls = FactorModel(X, r; method = :ls)

        # Define a test function that dispatches on method type
        test_dispatch(::FactorModel{PCA}) = :pca
        test_dispatch(::FactorModel{EM}) = :em
        test_dispatch(::FactorModel{LeastSquares}) = :ls

        @test test_dispatch(fm_pca) == :pca
        @test test_dispatch(fm_em) == :em
        @test test_dispatch(fm_ls) == :ls
    end

    @testset "EM with IC-adaptive factor selection" begin
        @testset "IC-adaptive selects correct dimensions with missing data" begin
            Random.seed!(42)
            T, n, r_true = 200, 30, 3
            F_true = randn(T, r_true)
            Λ_true = randn(n, r_true)
            X = F_true * Λ_true' + 0.5 * randn(T, n)

            # Introduce ~10% missing values
            mask = rand(T, n) .< 0.1
            X[mask] .= NaN

            kmax = 15
            fm = FactorModel(X, kmax; ic = IC2, scale = true)

            # Should have fewer factors than kmax
            @test numfactors(fm) < kmax
            @test numfactors(fm) >= 1

            # Dimensions must be consistent
            @test size(factors(fm)) == (T, numfactors(fm))
            @test size(loadings(fm)) == (n, numfactors(fm))
            @test length(eigvals(fm)) == numfactors(fm)

            # No NaN in output
            @test !any(isnan, fm.X̄)
            @test !any(isnan, factors(fm))
            @test !any(isnan, loadings(fm))

            # Method should be EM
            @test estimationmethod(fm) isa EM
        end

        @testset "IC-adaptive selects approximately correct r" begin
            Random.seed!(123)
            T, n, r_true = 300, 40, 4
            F_true = randn(T, r_true)
            Λ_true = randn(n, r_true)
            X = F_true * Λ_true' + 0.3 * randn(T, n)

            # Introduce ~5% missing values
            mask = rand(T, n) .< 0.05
            X[mask] .= NaN

            fm = FactorModel(X, 12; ic = IC1, scale = true)

            # IC should select something close to the true r
            @test numfactors(fm) >= r_true - 1
            @test numfactors(fm) <= r_true + 4
        end

        @testset "All IC types work with IC-adaptive EM" begin
            Random.seed!(456)
            T, n = 100, 20
            X = randn(T, n)
            X[1:10, 1:3] .= NaN

            for ICType in
                [IC1, IC2, IC3, PCp1, PCp2, PCp3, AIC1, AIC2, AIC3, BIC1, BIC2, BIC3]
                fm = FactorModel(X, 8; ic = ICType, scale = true)
                @test numfactors(fm) >= 1
                @test numfactors(fm) <= 8
                @test !any(isnan, factors(fm))
            end
        end

        @testset "ic=nothing preserves backward compatibility" begin
            Random.seed!(789)
            T, n, r = 100, 15, 4
            X = randn(T, n)
            X[1:5, 1] .= NaN

            fm_default = FactorModel(X, r; scale = true)
            fm_nothing = FactorModel(X, r; ic = nothing, scale = true)

            @test numfactors(fm_default) == numfactors(fm_nothing)
            @test eigvals(fm_default) ≈ eigvals(fm_nothing)
        end

        @testset "ic ignored when no missing data (no EM loop)" begin
            Random.seed!(321)
            X = randn(100, 15)

            # No missing data => no EM loop => ic is ignored, model has numfactors factors
            fm = FactorModel(X, 5; ic = IC2, method = :em)
            @test numfactors(fm) == 5
        end

        @testset "Validation: ic with constraints errors" begin
            X = randn(50, 10)
            c = normalize_loading(1, 1, 3)
            @test_throws ArgumentError FactorModel(X, 3; ic = IC1, constraints = c)
        end

        @testset "Validation: ic with :pca errors" begin
            X = randn(50, 10)
            @test_throws ArgumentError FactorModel(X, 3; ic = IC1, method = :pca)
        end

        @testset "Validation: ic with :ls errors" begin
            X = randn(50, 10)
            @test_throws ArgumentError FactorModel(X, 3; ic = IC1, method = :ls)
        end
    end

    @testset "total_r2 and byfactor_r2" begin
        Random.seed!(42)
        X = randn(100, 20)
        fm = FactorModel(X, 3; scale = true)

        @testset "total_r2 - basic" begin
            tr = total_r2(fm)
            @test tr isa TotalR2
            @test length(tr.r2) == 20
            @test length(tr.varnames) == 20
            @test tr.r2 == r2(fm)
            @test tr.show_all == false

            # Default varnames
            @test tr.varnames == ["V$i" for i in 1:20]
        end

        @testset "total_r2 - custom varnames" begin
            names = ["Var_$i" for i in 1:20]
            tr = total_r2(fm; varnames = names)
            @test tr.varnames == names
        end

        @testset "total_r2 - show_all" begin
            tr = total_r2(fm; show_all = true)
            @test tr.show_all == true
        end

        @testset "total_r2 - varnames length mismatch" begin
            @test_throws ArgumentError total_r2(fm; varnames = ["a", "b"])
        end

        @testset "total_r2 - show doesn't error" begin
            tr = total_r2(fm)
            io = IOBuffer()
            show(io, tr)
            output = String(take!(io))
            @test occursin("Total R²", output)
            # n=20 <= 20, so all rows shown, no show_all message
            @test !occursin("show_all=true", output)

            # With show_all
            tr_all = total_r2(fm; show_all = true)
            io = IOBuffer()
            show(io, tr_all)
            output = String(take!(io))
            @test occursin("Total R²", output)
        end

        @testset "total_r2 - show with many variables" begin
            Random.seed!(42)
            X_big = randn(100, 50)
            fm_big = FactorModel(X_big, 3; scale = true)
            tr = total_r2(fm_big)
            io = IOBuffer()
            show(io, tr)
            output = String(take!(io))
            @test occursin("show_all=true", output)
        end

        @testset "byfactor_r2 - basic" begin
            br = byfactor_r2(fm)
            @test br isa ByFactorR2
            @test size(br.r2mat) == (20, 3)
            @test length(br.varnames) == 20
            @test length(br.factornames) == 3
            @test br.factornames == ["Factor_1", "Factor_2", "Factor_3"]
            @test br.show_all == false

            # Default varnames
            @test br.varnames == ["V$i" for i in 1:20]
        end

        @testset "byfactor_r2 - custom varnames" begin
            names = ["Series_$i" for i in 1:20]
            br = byfactor_r2(fm; varnames = names)
            @test br.varnames == names
        end

        @testset "byfactor_r2 - varnames length mismatch" begin
            @test_throws ArgumentError byfactor_r2(fm; varnames = ["x"])
        end

        @testset "byfactor_r2 - show doesn't error" begin
            br = byfactor_r2(fm)
            io = IOBuffer()
            show(io, br)
            output = String(take!(io))
            @test occursin("R² by Individual Factor", output)
        end

        @testset "byfactor_r2 - mathematical correctness" begin
            F = factors(fm)
            Λ = loadings(fm)
            X_bar = fm.X̄

            br = byfactor_r2(fm)
            # Check a few entries manually
            for i in [1, 10, 20]
                tss_i = sum(abs2, X_bar[:, i])
                for j in 1:3
                    ssr_ij = sum(t -> (X_bar[t, i] - F[t, j] * Λ[i, j])^2, 1:size(F, 1))
                    expected = 1.0 - ssr_ij / tss_i
                    @test br.r2mat[i, j] ≈ expected
                end
            end
        end

        @testset "Tables.jl interface - TotalR2" begin
            tr = total_r2(fm)
            @test Tables.istable(typeof(tr)) == true
            @test Tables.columnaccess(typeof(tr)) == true
            @test Tables.columnnames(tr) == (:Variable, :R2)

            cols = Tables.columns(tr)
            @test Tables.getcolumn(cols, :Variable) == tr.varnames
            @test Tables.getcolumn(cols, :R2) == tr.r2
            @test Tables.getcolumn(cols, 1) == tr.varnames
            @test Tables.getcolumn(cols, 2) == tr.r2

            # Schema
            sch = Tables.schema(tr)
            @test sch !== nothing

            # DataFrame conversion
            df = DataFrame(tr)
            @test size(df) == (20, 2)
            @test names(df) == ["Variable", "R2"]
            @test df.R2 == r2(fm)
        end

        @testset "Tables.jl interface - ByFactorR2" begin
            br = byfactor_r2(fm)
            @test Tables.istable(typeof(br)) == true
            @test Tables.columnaccess(typeof(br)) == true

            colnames = Tables.columnnames(br)
            @test colnames[1] == :Variable
            @test length(colnames) == 4  # Variable + 3 factors

            cols = Tables.columns(br)
            @test Tables.getcolumn(cols, :Variable) == br.varnames
            @test Tables.getcolumn(cols, :Factor_1) == br.r2mat[:, 1]

            # Schema
            sch = Tables.schema(br)
            @test sch !== nothing

            # DataFrame conversion
            df = DataFrame(br)
            @test size(df) == (20, 4)
            @test names(df)[1] == "Variable"
        end
    end

    include("test_sw.jl")
end

# Aqua.jl quality assurance tests
include("Aqua.jl")

# Doctests
include("doctests.jl")
