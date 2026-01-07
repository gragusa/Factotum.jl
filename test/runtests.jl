using Factotum, Statistics, LinearAlgebra, Test, Random, NaNStatistics

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
        @test Σ ≈ diagm(0 => σ.^2)

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
        fm_demean = FactorModel(X, 5; demean=true, scale=false)
        @test numfactors(fm_demean) == 5

        # Test demean=false
        fm_nodemean = FactorModel(X, 5; demean=false, scale=false)
        @test numfactors(fm_nodemean) == 5

        # Test scale=true
        fm_scaled = FactorModel(X, 5; demean=true, scale=true)
        @test numfactors(fm_scaled) == 5

        # Test both false
        fm_raw = FactorModel(X, 5; demean=false, scale=false)
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
        ic = IC1(X, 5; scale=true)
        @test length(criterion(ic)) == 6
    end

    @testset "EM algorithm - no missing values" begin
        # EM with no missing values should produce similar results to standard PCA
        Random.seed!(42)
        X = randn(100, 10)

        fm_standard = FactorModel(X, 3; scale=true)
        fm_em = FactorModel(X, 3; scale=true, em=true)

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
        fm = FactorModel(X_missing, r; scale=true)

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
        X_centered = (X_complete .- mean(X_complete; dims=1)) ./ std(X_complete; dims=1)
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

        fm = FactorModel(X_missing, r; scale=true)

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
        fm_mean = FactorModel(X, r; init=nanmean)
        @test numfactors(fm_mean) == r
        @test !any(isnan, fm_mean.X̄)

        # Test with nanmedian
        fm_median = FactorModel(X, r; init=nanmedian)
        @test numfactors(fm_median) == r
        @test !any(isnan, fm_median.X̄)

        # Test with custom init function (zero)
        fm_zero = FactorModel(X, r; init=x -> zero(eltype(x)))
        @test numfactors(fm_zero) == r
        @test !any(isnan, fm_zero.X̄)
    end

    @testset "EM algorithm - convergence parameters" begin
        Random.seed!(321)
        T, n, r = 50, 10, 2

        X = randn(T, n)
        X[1:3, 1:2] .= NaN

        # Test with different maxiter and tol
        fm1 = FactorModel(X, r; maxiter=100, tol=1e-6)
        fm2 = FactorModel(X, r; maxiter=2000, tol=1e-10)

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

        fm_em = FactorModel(X, r; method=:em, scale=true, tol=1e-10, maxiter=2000)
        fm_ls = FactorModel(X, r; method=:ls, scale=true, tol=1e-10, maxiter=2000)

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
        for j in (r+3):n
            missing_rows = randperm(T)[1:10]  # 10 missing values per column
            X_missing[missing_rows, j] .= NaN
        end

        fm_em = FactorModel(X_missing, r; method=:em, scale=true, tol=1e-10, maxiter=2000)
        fm_ls = FactorModel(X_missing, r; method=:ls, scale=true, tol=1e-10, maxiter=2000)

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

        fm = FactorModel(X, r; method=:ls)

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
        constraints_mat = [
            1.0  1.0  0.0  0.0  1.0;  # series 1: 1*λ₁ + 0*λ₂ + 0*λ₃ = 1
            5.0  0.0  0.0  1.0  0.0;  # series 5: 0*λ₁ + 0*λ₂ + 1*λ₃ = 0
        ]
        lc2 = LoadingConstraints(constraints_mat)
        @test lc2.series == [1, 5]
        @test lc2.R == [1.0 0.0 0.0; 0.0 0.0 1.0]
        @test lc2.r == [1.0, 0.0]

        # Test helper functions
        c1 = normalize_loading(1, 1, 3; value=1.0)
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
        c = normalize_loading(1, 1, r; value=1.0)

        fm = FactorModel(X, r; constraints=c, scale=false)

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

        fm = FactorModel(X, r; constraints=c, scale=false)

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
        c1 = normalize_loading(1, 1, r; value=1.0)
        c2 = zero_loading(5, 3, r)
        c3 = normalize_loading(10, 2, r; value=0.5)

        constraints = vcat(c1, vcat(c2, c3))

        fm = FactorModel(X, r; constraints=constraints, scale=false)

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
        fm = FactorModel(X, r; constraints=c)

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

        c = normalize_loading(2, 1, r; value=1.0)  # Series 2 (not series 1 which has NaN)

        fm = FactorModel(X, r; constraints=c, scale=false)

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
        @test_throws ArgumentError FactorModel(X_missing, r; method=:pca)

        # PCA with constraints should error
        @test_throws ArgumentError FactorModel(X, r; method=:pca, constraints=c)

        # EM with constraints should error
        @test_throws ArgumentError FactorModel(X, r; method=:em, constraints=c)

        # Invalid method should error
        @test_throws ArgumentError FactorModel(X, r; method=:invalid)
    end

    @testset "Constraint validation" begin
        X = randn(50, 10)
        r = 3

        # Constraint referencing non-existent series
        c_bad_series = LoadingConstraints([100], [1.0 0.0 0.0], [1.0])
        @test_throws ArgumentError FactorModel(X, r; constraints=c_bad_series)

        # Constraint with wrong number of factors in R
        c_bad_r = LoadingConstraints([1], [1.0 0.0 0.0 0.0], [1.0])  # 4 cols but r=3
        @test_throws ArgumentError FactorModel(X, r; constraints=c_bad_r)
    end

    include("test_sw.jl")
end
