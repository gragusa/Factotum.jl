module Factotum

using IterTools
using Distributions
using StatsBase
using Optim
using DataFrames
using Printf
using Statistics
using LinearAlgebra
import StatsBase: residuals, fit

function staticfactor(Z; demean::Bool = true, scale::Bool = false)
    ## Estimate
    ## X = FΛ' + e
    ##
    ## Λ'Λ = V, V (rxr) diagonal
    ## F'F = I
    T, n = size(Z)
    μ = demean ? mean(Z, dims=1) : zeros(1,n)
    σ = scale ? std(Z, dims=1) : ones(1,n)
    X = (Z .- μ)./σ
    ev = eigen(X'*X)
    neg = findall(x -> x < 0, ev.values)
    if !isempty(neg)
        if any(ev.values[neg] .< -9 * eps(Float64) * first(ev.values))
            error("covariance matrix is not non-negative definite")
        else
            ev.values[neg] = 0.0
        end
    end
    ## Stored in reverse order
    λ  = ev.values[n:-1:1]
    σ  = sqrt.(λ/T)
    Vₖ = (σ.^2)./sum(σ.^2)
    Λ = sqrt(n).*ev.vectors[:, n:-1:1]
    F = X*Λ/n
    (F, Λ, λ, σ, Vₖ, X, n, μ, σ)
end

struct FactorModel
    factors::Array{Float64, 2}
    loadings::Array{Float64, 2}
    eigenvalues::Array{Float64, 1}
    sdev::Array{Float64, 1}
    explained_variance::Array{Float64, 1}
    residuals::Matrix{Float64}
    residual_variance::Float64
    center::Array{Float64, 2}
    scale::Array{Float64, 1}
    k::Int64
    X::Matrix{Float64}
    r::Matrix{Float64}
end

abstract type SelectionCriteria end


struct ICp1 <: SelectionCriteria end
struct ICp2 <: SelectionCriteria end
struct ICp3 <: SelectionCriteria end

struct PCp1 <: SelectionCriteria end
struct PCp2 <: SelectionCriteria end
struct PCp3 <: SelectionCriteria end

struct AIC1 <: SelectionCriteria end
struct AIC2 <: SelectionCriteria end
struct AIC3 <: SelectionCriteria end

struct BIC1 <: SelectionCriteria end
struct BIC2 <: SelectionCriteria end
struct BIC3 <: SelectionCriteria end


# function calculate_residual(fm::FactorModel, k)
#     T, N = size(fm.X)
#     F = view(fm.factors, :, 1:k)
#     ## X = fm.X
#     ## Λ = F\X
#     Λ = view(fm.loadings, :, 1:k)
#     (fm.X .- F*Λ')
# end

# function calculate_residual_variance(fm::FactorModel, k)
#     T, N = size(fm.X)
#     F = view(fm.factors, :, 1:k)
#     ## X = fm.X
#     ## Λ = F\X
#     Λ = view(fm.loadings, :, 1:k)
#     fm.r .= (fm.X .- F*Λ')
#     sum(fm.r.^2)/(T*N)
# end

function FactorModel(Z::Matrix{Float64}; kwargs...)
    (factors,
     loadings,
     eigenvalues,
     sdev,
     explained_variance,
     X,
     k,
     center,
     scale) = staticfactor(Z; kwargs...)

    T, n = size(X)
    residuals = (X .- factors*loadings').^2
    residual_variance = sum(residuals)/(T*n)
    FactorModel(factors,
                loadings,
                eigenvalues,
                sdev,
                explained_variance,
                residuals,
                residual_variance,
                center,
                scale,
                size(factors, 2),
                copy(X),
                similar(X))
end

numfactor(fm::FactorModel) = fm.k

function factors(fm::FactorModel, k)
    @assert k <= numfactor(fm) "k too large"
    fm.factors[:, 1:k]
end

function loadings(fm::FactorModel, k)
    @assert k <= numfactor(fm) "k too large"
    fm.loadings[:, 1:k]
end

function eigenvalues(fm::FactorModel, k)
    @assert k <= numfactor(fm) "k too large"
    fm.eigenvalues[1:k]
end

function sdev(fm::FactorModel, k)
    @assert k <= numfactor(fm) "k too large"
    fm.sdev[1:k]
end

function StatsBase.residuals(fm::FactorModel, k)
    @assert k <= numfactor(fm) "k too large"
    fm.residuals[:, 1:k]
end

function residual_variance(fm::FactorModel, k)
    @assert k <= numfactor(fm) "k too large"
    mean((fm.X - factors(fm, k)*loadings(fm, k)').^2)
end


factors(fm::FactorModel) = factors(fm, fm.k)
loadings(fm::FactorModel) = loadings(fm, fm.k)
eigenvalues(fm::FactorModel) = eigenvalues(fm, fm.k)
residual_variance(fm::FactorModel) = residual_variance(fm, fm.k)
sdev(fm::FactorModel) = sdev(fm, fm.k)

function subview(fm::FactorModel, k)
    T, n = size(fm.X)
    fac = factors(fm, k)
    lod = loadings(fm, k)
    eig = eigenvalues(fm, k)
    sdv = sdev(fm, k)
    exv = sdv.^2/sum(sdv.^2)
    res = residuals(fm, k)
    rsv = mean((fm.X .- fac*lod').^2)

    FactorModel(fac, ## factors
                lod, ## loadings
                eig, ## eigenvalues
                sdv, ## sdev
                exv, ## explained_variance
                res, ## residuals
                rsv, ## residual_variance
                fm.center,
                fm.scale,
                k,
                fm.X,
                fm.r)
end

function Base.show(io::IO, fm::FactorModel)
    printstyled(io, "\nStatic Factor Model\n", color = :green)
    #@printf io "------------------------------------------------------\n"
    @printf io "Dimensions of X..........: %s\n" size(fm.X)
    @printf io "Number of factors........: %s\n" fm.k
    @printf io "Factors calculated by....: %s\n" "Principal Component"
    @printf io "Residual variance........: %s\n" residual_variance(fm)
    #@printf io "\n"
    #@printf io "------------------------------------------------------\n"
end

function factortable(io::IO, fm::FactorModel)
    colnms = "Factor_".*string.(1:numfactor(fm))
    rownms = ["Standard deviation", "Proportion of Variance", "Cumulative Proportion"]
    mat = vcat(sdev(fm)', fm.explained_variance', cumsum(fm.explained_variance)')
    ct = CoefTable(mat, colnms, rownms)
    show(io, ct)
end

describe(fm::FactorModel) = describe(stdout, fm)

function describe(io::IO, fm::FactorModel)
    show(io, fm)
    printstyled(io, "Factors' importance:\n", color = :green)
    factortable(io, fm)
end

struct Criteria{M <: SelectionCriteria}
    criterion::M
    crit::Array{Float64,1}
    kmax::UnitRange{Int64}
end

variance_factor(::Type{M}, fm, kmax) where M <: Union{ICp1, ICp2, ICp3} = 1.0
variance_factor(::Type{M}, fm, kmax) where M <: SelectionCriteria = residual_variance(fm, kmax)

transform_V(::Type{M}, V) where M <: Union{ICp1, ICp2, ICp3} = log.(V)
transform_V(::Type{M}, V) where M <: SelectionCriteria = V

function Criteria(s::Type{M}, fm::FactorModel, kmax::Int64) where M <: SelectionCriteria
    T, n = size(fm.X)
    rnge = 0:kmax
    σ̂² = variance_factor(M, fm, kmax)
    models = map(k -> subview(fm, k), rnge)
    Vₖ = map(x -> residual_variance(x), models)
    Vₖ = transform_V(M, Vₖ)
    gₜₙ= map(k -> penalty(s, T, n, k), rnge)
    Criteria(M(), Vₖ + σ̂².*(rnge).*gₜₙ, rnge)
end

function Base.show(io::IO, c::T) where T <: Criteria
    mat = c.crit[:,:]
    rownms = "k = ".*string.(collect(c.kmax))
    colnms = [string(c.criterion)]
    ct = StatsBase.CoefTable(mat, colnms, rownms)
    show(io, ct)
end

function penalty(s::Type{P}, T, N, k) where P <: Union{ICp1, PCp1}
    NtT = N*T
    NpT = N+T
    p1 = NpT/NtT
    p2 = log(NtT/NpT)
    p1*p2
end

function penalty(s::Type{P}, T, N, k) where P <: Union{ICp2, PCp2}
    C2  = min(T, N)
    NtT = N*T
    NpT = N+T
    p1 = NpT/NtT
    p2 = log(C2)
    p1*p2
end

function penalty(s::Type{P}, T, N, k) where P <: Union{ICp3, PCp3}
    C2  = min(T, N)
    log(C2)/C2
end

penalty(s::Type{AIC1}, T, N, k) = 2/T
penalty(s::Type{AIC2}, T, N, k) = 2/N
penalty(s::Type{AIC3}, T, N, k) = 2*(N+T-k)/(N*T)

penalty(s::Type{BIC1}, T, N, k) = log(T)/T
penalty(s::Type{BIC2}, T, N, k) = log(N)/N
penalty(s::Type{BIC3}, T, N, k) = 2*((N+T-k)*log(N*T))/(N*T)


struct WaldTest
    tbl::DataFrame
    rankmin::Int64
    rankmax::Int64
end

struct WaldTestFun{F, T, Z}
    f::F
    r::Int64
    vecsigma::T
    Vhat::Z
end

(wf::WaldTestFun)(theta) = wf.f(theta, wf.r, wf.vecsigma, wf.Vhat)

function waldobjfun(th, r, vecsigma, Vhat)
    ##r,k = size(theta) ## note that the rank being tested is r0 = r-1
    theta = reshape(th, r+1, length(th)÷(r+1))
    sigmamat = diagm(0=>theta[1,:].^2) .+ theta[2:r+1,:]'*theta[2:r+1,:]
    tempsigma = sigmamat[findall(tril(ones(size(sigmamat))).==1)]
    (vecsigma -tempsigma)' /Vhat *(vecsigma - tempsigma)
end

X = randn(100,10);
fm = Factotum.FactorModel(X)

function waldtest(fm::FactorModel, minrank::Int = 0, maxrank::Int = 2)
    X = copy(fm.X)
    T, n = size(X)
    ## Normalize factor
    Xs = X / diagm(0=>sqrt.(diag(cov(X))))
    covX = cov(Xs)
    meanX = mean(Xs, dims=1)
    vecsigma = Factotum.vech(covX)
    bigN = length(vecsigma)
    Vhat = Array{Float64}(undef, bigN, bigN)
    varvecsig = zeros(n,n,n,n);

    for i1 in 1:n, i2 = 1:n, i3 = 1:n, i4 = 1:n
        varvecsig[i1,i2,i3,i4] = sum( (Xs[:,i1] .- meanX[i1]).*(Xs[:,i2] .- meanX[i2]).*(Xs[:,i3] .- meanX[i3]) .*(Xs[:,i4] .- meanX[i4])) / T^2 - covX[i1,i2] *covX[i3,i4] /T
    end

    idx = findall(tril(ones(size(covX))).==1)
    for i=1:bigN, j=1:bigN   ## map elements of varvecsig array into matrix corresponding to
        Vhat[i,j] = varvecsig[idx[i],idx[j]]
    end

    out_table = DataFrame(rank = -1, waldstat = NaN, df = NaN, critval = NaN, pvalue = NaN)

    ## Initial values
    for k in minrank:maxrank
        wf = WaldTestFun(waldobjfun, k, vecsigma, Vhat)
        df = (n-k)*(n-k+1)/2 - n

        theta0 = theta_initial_value(n,k)

        outs = Array{Tuple{Float64,Array{Float64,1},Bool},1}()

        for j in theta0
            out = Optim.optimize(wf, j, BFGS(), Optim.Options(allow_f_increases=true); autodiff=:forward)
            push!(outs, (out.minimum::Float64, out.minimizer::Array{Float64,1}, Optim.converged(out)::Bool))
        end

        convouts = outs[map(x->x[3], outs)]
        out      = convouts[argmin(map(x->x[1], convouts))]

        chisq = Chisq(df)
        dfa = DataFrame(rank = k, waldstat = out[1], df = df, critval = Distributions.quantile(chisq, .95), pvalue = 1-Distributions.cdf(chisq, out[1]))
        append!(out_table, dfa)
    end
    filter!(raw->raw[:rank]>=0, out_table)
    WaldTest(out_table, minrank, maxrank)
end

function theta_initial_value(n,k)
    I3 = ones(1,n)/3
    ek = [diagm(0=>ones(k)) zeros(k, n-k)]
    t0 = ([I3; zeros(k,n)], [I3; ones(k,n)./(2*k)],[I3; ek./(2*k)], [I3; reverse(ek./(2*k), dims=2)])::NTuple{4,Array{Float64,2}}
    map(vec, t0)::NTuple{4,Array{Float64,1}}
end



function vech(X::Matrix{S}) where S
    T, n = size(X)
    r = round(Int64, n*(n+1)/2)
    x = Array{S, 1}(undef, r)
    i = 1
    for j in 1:n, k in j:n
        x[i] = X[j,k]
        i += 1
    end
    x
end


export FactorModel, subview, waldtest, describe, PValue, waldstat, Criteria,
       ICp1, ICp2, ICp3, PCp1, PCp2, PCp3,
       AIC1, AIC2, AIC3, BIC1, BIC2, BIC3




end # module
