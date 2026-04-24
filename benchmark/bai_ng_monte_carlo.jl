#!/usr/bin/env julia

"""
Monte Carlo replication of the static-factor DGP in Bai & Ng (2002), Section 6.

DGP:
    X_it = sum_{j=1}^r λ_ij F_tj + sqrt(θ*r) e_it
with λ_ij, F_tj, e_it ~ i.i.d. N(0,1).

This script evaluates factor-number criteria implemented in Factotum.jl and reports:
- mean selected number of factors
- hit rate (Pr(\hat r = r_true))
"""

using Factotum
using Printf
using Random
using Statistics

const DEFAULT_CRITERIA = (
    IC1, IC2, IC3,
    PCp1, PCp2, PCp3,
    AIC1, AIC2, AIC3,
    BIC1, BIC2, BIC3,
)

"""
    simulate_bai_ng_dgp(rng, T, N, r; θ=1.0)

Generate a T × N panel from the Bai-Ng Gaussian benchmark DGP.
"""
function simulate_bai_ng_dgp(
        rng::AbstractRNG,
        T::Int,
        N::Int,
        r::Int;
        θ::Float64 = 1.0,
    )
    F = randn(rng, T, r)
    Λ = randn(rng, N, r)
    ε = randn(rng, T, N)
    F * Λ' .+ sqrt(θ * r) .* ε
end

"""
    monte_carlo_bai_ng(; kwargs...)

Run Monte Carlo for Bai-Ng DGP and compute criterion selection summaries.
"""
function monte_carlo_bai_ng(
        ;
        reps::Int = 1000,
        T::Int = 60,
        N::Int = 100,
        r_true::Int = 1,
        kmax::Int = 8,
        θ::Float64 = 1.0,
        seed::Int = 20260325,
        scale::Bool = false,
        criteria::Tuple = DEFAULT_CRITERIA,
    )
    reps > 0 || throw(ArgumentError("reps must be positive"))
    0 < r_true <= kmax || throw(ArgumentError("r_true must satisfy 0 < r_true <= kmax"))

    rng = MersenneTwister(seed)
    selected = Dict{DataType, Vector{Int}}(c => Int[] for c in criteria)

    for _ in 1:reps
        X = simulate_bai_ng_dgp(rng, T, N, r_true; θ = θ)
        fm = FactorModel(X, kmax; scale = scale)
        for C in criteria
            push!(selected[C], numfactors(C(fm, kmax)))
        end
    end

    summary = Dict{DataType, NamedTuple{(:mean_r, :hit_rate), Tuple{Float64, Float64}}}()
    for C in criteria
        rs = selected[C]
        summary[C] = (
            mean_r = mean(rs),
            hit_rate = mean(==(r_true), rs),
        )
    end

    return (
        config = (reps = reps, T = T, N = N, r_true = r_true, kmax = kmax, θ = θ, scale = scale,
            seed = seed),
        summary = summary,
        selections = selected,
    )
end

function print_summary_table(result)
    cfg = result.config
    println("Bai-Ng Monte Carlo")
    println("-------------------")
    println("reps=$(cfg.reps), T=$(cfg.T), N=$(cfg.N), r=$(cfg.r_true), kmax=$(cfg.kmax), θ=$(cfg.θ), scale=$(cfg.scale)")
    println()
    println(rpad("Criterion", 8), "  ", lpad("E[r̂]", 8), "  ", lpad("Pr(r̂=r)", 10))
    for C in DEFAULT_CRITERIA
        s = result.summary[C]
        @printf("%-8s  %8.3f  %10.3f\n", string(C), s.mean_r, s.hit_rate)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    reps = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1000
    T = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 60
    N = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 100
    r_true = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 1
    kmax = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 8

    result = monte_carlo_bai_ng(; reps = reps, T = T, N = N, r_true = r_true, kmax = kmax)
    print_summary_table(result)
end
