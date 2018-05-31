module Factotum

using IterTools
using Distributions 
using PDMats
using StatsBase
using NLPModels
using Ipopt
using MathProgBase
import StatsBase: residuals, fit

function staticfactor(Z; demean::Bool = true, scale::Bool = false)
    ## Estimate
    ## X = FΛ' + e
    ##
    ## Λ'Λ = V, V (rxr) diagonal
    ## F'F = I
    T, n = size(Z)
    μ = demean ? mean(Z, 1) : zeros(1,n)
    σ = scale ? std(Z, 1) : ones(1,n)
    X = (Z .- μ)./σ
    ev = eigfact(X'*X)
    neg = find(x -> x < 0, ev.values)
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
    Vₖ = σ.^2./sum(σ.^2) 
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
    print_with_color(:green, io, "\nStatic Factor Model\n")
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

describe(fm::FactorModel) = describe(STDOUT::IO, fm)

function describe(io::IO, fm::FactorModel)
    show(io, fm)
    print_with_color(:green, io, "Factors' importance:\n")
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
    W::Float64
    pvalue::Float64
    knull::Int64
    status::Symbol
    solver::MathProgBase.SolverInterface.AbstractMathProgSolver
end

function waldtest(fm::FactorModel, knull::Int64; solver = IpoptSolver())
    X = fm.X
    T, n = size(X)
    df = n*(n+1)/2 - (n*knull + n) + knull*(knull-1)/2

    @assert df > 0 "Cannot perform wald test. df <= 0"

    Λ  = Factotum.loadings(fm, knull)
    F  = Factotum.factors(fm, knull)
    Σₓ = Factotum.vech(X'X/T)
    #Ω  = PDMat(Factotum.calculate_variance_of_xx(X))
    Ω   = pinv(Factotum.calculate_variance_of_xx(X))
    function fobj(parms)
        ## Parms are n*k0 (\Lambda) e n variances of 
        Λ = reshape(parms[1:n*knull], n, knull)
        ϵ = diagm(parms[n*knull+1:end])
        v = Σₓ - Factotum.vech(Λ*Λ' + ϵ)
        v'*Ω*v
    end

    x0 =  [ vec(Λ); vec(diag((X .- F*Λ')'*(X .- F*Λ')/T)) ]

    nlp = ADNLPModel(fobj, x0, 
                lvar = [repeat([-Inf], outer = n*knull); repeat([0.0000001], outer = n)],
                uvar = [repeat([+Inf], outer = n*knull); repeat([+Inf], outer = n)])

    mb = NLPtoMPB(nlp, solver)

    MathProgBase.optimize!(mb)

    x  = MathProgBase.SolverInterface.getsolution(mb)
    W  = MathProgBase.SolverInterface.getobjval(mb)
    status = MathProgBase.SolverInterface.status(mb)

    if status != :Optimal
        warn("The optimazation did not converge. Use the results with care")
    end

    pvalue = 1-cdf(Chisq(df), W)
    WaldTest(W, pvalue, knull, status, solver)
end

StatsBase.PValue(w::WaldTest) = w.pvalue
waldstat(w::WaldTest) = w.W

function Base.show(io::IO, w::WaldTest)
    colnms = ["Wald stat", " Pr(>W)"]
    mat = [w.W w.pvalue]
    rownms = [""]
    ct = StatsBase.CoefTable(mat, colnms, rownms, 2)
    @printf io "Wald test [Null:] H₀:k=%s\n" w.knull
    show(io, ct)
end

function calculate_variance_of_xx(X::Array{S,2}) where S <: Real
    ## X has mean zeros along dimension 1
    T, n = size(X)
    σₓ = X'X/T
    
    itr = (((k,j) for j in 1:n for k in j:n))    
    r  = round(Int64, n*(n+1)/2)
    V = zeros(r^2)
    i = 1
    for ((k,j),(l,m)) in product(itr,itr)
        for t = 1:T
            V[i] += (X[t,j]*X[t,k]-σₓ[j,k])*(X[t,l]*X[t,m]-σₓ[l,m])
        end
        i += 1
    end
    reshape(V./T^2, r, r)
end

function vech(X::Matrix{S}) where S
    T, n = size(X)
    r = round(Int64, n*(n+1)/2)
    x = Array{S, 1}(r)
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
