@testset "Stock-Watson LS factor model" begin
    using CSV, DataFrames, NaNStatistics

    path = joinpath(pkgdir(Factotum), "test", "data", "macrodata.csv")
    df = CSV.read(path, DataFrame)
    X_full = Matrix{Float64}(df[:, 2:end])

    ftm = Factotum.FactorModel(X_full, 6; scale=true, method=:ls)

    expected_factors = [
        -1.8682   -0.3544   -0.2306   -0.0071    1.7681   -0.9233
        -4.4440   -0.1107    0.1187   -0.2745   -0.2588   -0.8757
        -2.6460   -0.0584   -1.5195    0.5259   -0.6978   -1.1627
        -2.4535   -0.6135   -6.6660    1.2342   -0.2477   -1.4134
    ]

    expected_loadings = [
        -0.00892726   -0.00608655    0.0435287    -0.00612441   -0.00356542   -0.00157926
        -0.0083041    -0.00316742    0.0400376    -0.00892476   -0.0118924    -0.00684809
        -0.00446497   -0.0032679     0.0444272    -0.0101491    -0.00240275   -0.00959379
        -0.0032462    -0.000259264   0.0240073    -0.00533251   -0.00374463   -0.00709906
    ]

    @testset "Factor values (last 4 rows)" begin
        @test ftm.factors[end-3:end, :] ≈ expected_factors atol=1e-3
    end

    @testset "Loading values (last 4 rows)" begin
        @test ftm.loadings[end-3:end, :] ≈ expected_loadings atol=1e-5
    end
end
