using Factotum, Statistics, LinearAlgebra, Test, Random

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

        # Check Λ'Λ/n = I
        @test Λ'Λ / n ≈ diagm(0 => ones(r))

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

        @test length(ic1.crit) == 11  # 0 to 10 factors
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
            @test length(ic.crit) == 6  # 0 to 5 factors
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
        describe(io, fm)
        output = String(take!(io))
        @test occursin("Static Factor Model", output)
    end

    @testset "Criterion from matrix directly" begin
        X = randn(100, 10)
        ic = IC1(X, 5; scale=true)
        @test length(ic.crit) == 6
    end

end
