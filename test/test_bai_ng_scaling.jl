using Factotum
using Random
using Statistics
using Test

# Bai-Ng DGP used in Econometrica (2002), Section 6
function simulate_bai_ng_dgp(rng::AbstractRNG, T::Int, N::Int, r::Int; θ::Float64 = 1.0)
    F = randn(rng, T, r)
    Λ = randn(rng, N, r)
    ε = randn(rng, T, N)
    F * Λ' .+ sqrt(θ * r) .* ε
end

function residual_variance_path(fm::FactorModel, kmax::Int)
    x = fm.X̄
    v0 = sum(abs2, x) / length(x)
    vk = [sum(abs2, residuals(view(fm, k))) / length(x) for k in 1:kmax]
    [v0; vk]
end

@testset "Bai-Ng IC/PCp scaling" begin
    Random.seed!(20260325)
    X = simulate_bai_ng_dgp(MersenneTwister(1), 80, 60, 2)
    kmax = 8
    fm = FactorModel(X, kmax; scale = false)

    V = residual_variance_path(fm, kmax)
    T, N = size(fm)

    # IC1: log(V(k)) + k*g1(N,T)
    g1 = [k * Factotum.penalty(IC1, T, N, k) for k in 0:kmax]
    ic1_manual = log.(V) .+ g1
    @test criterion(IC1(fm, kmax))≈ic1_manual rtol=1e-10 atol=1e-10

    # PCp1: V(k) + V(kmax)*k*g1(N,T)
    σ2 = V[end]
    pcp1_manual = V .+ σ2 .* g1
    @test criterion(PCp1(fm, kmax))≈pcp1_manual rtol=1e-10 atol=1e-10

    # Same check for IC2/IC3 and PCp2/PCp3
    g2 = [k * Factotum.penalty(IC2, T, N, k) for k in 0:kmax]
    g3 = [k * Factotum.penalty(IC3, T, N, k) for k in 0:kmax]

    @test criterion(IC2(fm, kmax))≈log.(V) .+ g2 rtol=1e-10 atol=1e-10
    @test criterion(IC3(fm, kmax))≈log.(V) .+ g3 rtol=1e-10 atol=1e-10

    @test criterion(PCp2(fm, kmax))≈V .+ σ2 .* g2 rtol=1e-10 atol=1e-10
    @test criterion(PCp3(fm, kmax))≈V .+ σ2 .* g3 rtol=1e-10 atol=1e-10
end

@testset "Bai-Ng Monte Carlo smoke test" begin
    Random.seed!(20260325)

    reps = 100
    T, N, r_true, kmax = 60, 100, 1, 8
    picks_ic1 = Int[]
    picks_ic2 = Int[]
    picks_ic3 = Int[]

    rng = MersenneTwister(42)
    for _ in 1:reps
        X = simulate_bai_ng_dgp(rng, T, N, r_true)
        fm = FactorModel(X, kmax; scale = false)
        push!(picks_ic1, numfactors(IC1(fm, kmax)))
        push!(picks_ic2, numfactors(IC2(fm, kmax)))
        push!(picks_ic3, numfactors(IC3(fm, kmax)))
    end

    # In Bai-Ng's baseline designs, IC criteria should center near the true r.
    @test abs(mean(picks_ic1) - r_true) <= 0.75
    @test abs(mean(picks_ic2) - r_true) <= 0.75
    @test abs(mean(picks_ic3) - r_true) <= 0.75
end
