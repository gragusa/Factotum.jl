module Factotum

using LinearAlgebra
using Optim
using PrettyTables
using Printf
using Statistics
using StatsBase
using StatsFuns

abstract type AbstractFactorModel end

struct FactorModel{M <: AbstractMatrix, V <: AbstractVector} <: AbstractFactorModel
    "The matrix of factors"
    factors::M
    "The matrix of loadings"
    loadings::M
    "The eigenvalues of X'X"
    eigenvalues::V
    "Residual " # Why here? Maybe to use to view and update in place...
    residuals::M
    "To demean and scale the matrix X"
    center::M
    scale::M
    "The original matrix"
    X::M
    "The rescaled matrix"
    X̄::M
end

struct FactorModelView{M <: AbstractMatrix, S <: AbstractMatrix, V <: AbstractVector, R} <: AbstractFactorModel
    "The matrix of factors"
    factors::M
    "The matrix of loadings"
    loadings::S
    "The eigenvalues of X'X"
    eigenvalues::V
    "Residual " # Why here? Maybe to use to view and update in place...
    residuals::R
    "The rescaled matrix"
    X̄::R
end

function FactorModel(Z::AbstractMatrix{G}, numfactors; kwargs...) where G
    T, n = size(Z)
    (F, Λ, λ, ε, μ, σₓ, Z, X) = T > n ? extract_ΛΛ(Z, numfactors; kwargs...) : extract_FF(Z; kwargs...)
    FactorModel(F, Λ, λ, ε, μ, σₓ, Z, X)
end

function extract_ΛΛ(Z, numfactors; demean::Bool = true, scale::Bool = false, corrected::Bool = false)
    ## Estimate
    ## X = FΛ' + e
    ##
    ## Λ'Λ = I
    ## F'F = V, V (rxr) diagonal
    T, n = size(Z)
    μ = demean ? mean(Z; dims = 1) : zeros(1, n)
    σₓ = scale ? std(Z; dims = 1, corrected = corrected) : ones(1, n)
    X = (Z .- μ)./σₓ
    ev = eigen(Symmetric(X'*X), n-numfactors+1:n)
    neg = findall(x -> x < 0, ev.values)
    if !isempty(neg)
        if any(ev.values[neg] .< -9 * eps(Float64) * first(ev.values))
            error("covariance matrix is not non-negative definite")
        else
            ev.values[neg] = 0.0
        end
    end
    λ  = ev.values[numfactors:-1:1] 
    Λ = sqrt(n)*ev.vectors[:, numfactors:-1:1]
    F = X*Λ/n
    ε = (X .- F*Λ')
    (F, Λ, λ, ε, μ, σₓ, Z, X)
end


function Base.view(fm::FactorModel, k::Int) 
    @assert k > 0 "Cannot view a FactorModel with 0 factors"
    view(fm, 1:k) 
end

function Base.view(fm::FactorModel, rnge::UnitRange)
    T, n = size(fm)
    @assert numfactors(fm) >= maximum(rnge) "Cannot create a FactorModel's view with $(maximum(rnge)) when the parent has $(numfactors(fm)) factors"
    @assert first(rnge) <= maximum(rnge) 
    ## To do: check that max(range) < numfactors fm
    FactorModelView(view(factors(fm), :, rnge), view(loadings(fm), :, rnge),
        view(eigvals(fm), rnge), residuals(fm), fm.X̄)
end


## ------------------------------------------------------------
## Methods
## ------------------------------------------------------------
Base.size(fm::AbstractFactorModel) = size(fm.X̄)
numfactors(fm::AbstractFactorModel) = size(loadings(fm), 2)
loadings(fm::AbstractFactorModel) = fm.loadings
factors(fm::AbstractFactorModel) = fm.factors

LinearAlgebra.eigvals(fm::AbstractFactorModel) = fm.eigenvalues

function sdev(fm::AbstractFactorModel) 
    T, n = size(fm)
    λ = eigvals(fm)
    sqrt.(λ/(T*n))
end

function explained_variance(fm::FactorModel) 
    λ = eigvals(fm)
    λ./sum(λ)
end

function StatsBase.residuals(fm::AbstractFactorModel)
    F = factors(fm)
    Λ = loadings(fm)
    fm.residuals .= fm.X̄ .- F*Λ'
    return fm.residuals
end

X(fm::FactorModel) = fm.X
X̄(fm::AbstractFactorModel) = fm.X̄

## Output
function Base.show(io::IO, fm::AbstractFactorModel)
    printstyled(io, "\nStatic Factor Model\n", color = :green)
    #@printf io "------------------------------------------------------\n"
    @printf io "Dimensions of X..........: %s\n" size(fm)
    @printf io "Number of factors........: %s\n" numfactors(fm)
    @printf io "Factors calculated by....: %s\n" "Principal Component"
    #@printf io "\n"
    #@printf io "------------------------------------------------------\n"
end

describe(fm::FactorModel) = describe(stdout, fm)

function describe(io::IO, fm::FactorModel)
    show(io, fm)
    printstyled(io, "Factors' importance:\n", color = :green)
    factortable(io, fm)
end

function factortable(io::IO, fm::FactorModel)
    explainedvar = explained_variance(fm)
    colnms = "Factor_".*string.(1:numfactors(fm))
    rownms = ["Standard deviation", "Proportion of Variance", "Cumulative Proportion"]
    mat = vcat(sdev(fm)', explainedvar', cumsum(explainedvar)')
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

struct InformationCriterion{M <: AbstractInformationCriterion, T<:AbstractFloat}
    criterion::M
    crit::Array{T,1}
    rnge::UnitRange{Int64} ## Change name of this field -> rnge
end

## Calculate V(F̂ᵏ) for k ⩽ kₘₐₓ 
function V(fmv::FactorModelView)
    ε = residuals(fmv)
    Λ = Factotum.loadings(fmv)
    F = Factotum.factors(fmv)
    mean(ε.^2)
end

V(fm::FactorModel, kₘₐₓ) = [V(view(fm, j)) for j ∈ 1:kₘₐₓ]    

variance_factor(::Type{M}, fm, kₘₐₓ) where M <: Union{IC1, IC2, IC3} = 1.0
variance_factor(::Type{M}, fm, kₘₐₓ) where M <: AbstractInformationCriterion = V(view(fm, kₘₐₓ)) ## This is probably transform
transform_V(::Type{M}, V) where M <: Union{IC1, IC2, IC3} = log.(V)
transform_V(::Type{M}, V) where M <: AbstractInformationCriterion = V

function informationcriterion(s::Type{M}, fm::FactorModel, kₘₐₓ::Int64) where M <: AbstractInformationCriterion
    T, n = size(fm)
    rnge = 1:kₘₐₓ
    σ̂²  = variance_factor(s, fm, kₘₐₓ)
    Vₖ = transform_V.(s, V(fm, kₘₐₓ))
    gₜₙ = map(k -> k*penalty(s, T, n, k), rnge)
    InformationCriterion(M(), Vₖ + σ̂².*gₜₙ, rnge)
end

function informationcriteria(criterion::Tuple, fm, kₘₐₓ)
    @assert all(map(x->isa(x(), Factotum.AbstractInformationCriterion), criterion)) "Some of the arguments is not a SelectionCriterion"
    map(x->x(fm, kₘₐₓ), criterion)
end

(for criterion ∈ (:BIC1, :BIC2, :BIC3, :AIC1, :AIC2, :AIC3, :IC1, :IC2, :IC3, :PCp1, :PCp2, :PCp3)
    eval(quote 
    ($criterion)(fm, kₘₐₓ::Int64) = Factotum.informationcriterion($criterion, fm, kₘₐₓ)
    Base.string(ic::($criterion)) = string(($criterion))
    ($criterion)(X::Matrix, kₘₐₓ::Int64; kwargs...) = Factotum.informationcriterion($criterion, FactorModel(X, kₘₐₓ; kwargs...), kₘₐₓ)
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
    map((nm,x, T) -> NamedTuple{nm}(x), nm, fmin, TT)
end

numfactors(ic::InformationCriterion) = findmin(ic).r
Base.string(ic::InformationCriterion{T, F}) where {T, F}  = string(T)

function Base.show(io::IO, ic::T) where T <: InformationCriterion
    header = ["# of factor"  "Criterion"]
    highlight1 = Highlighter((data, i, j) -> data[i,2] == minimum(data[:,2]), Crayon(background = :light_blue, foreground = :white, bold = true))
    highlight2 = Highlighter((data, i, j) -> (j == 2), Crayon(foreground = :light_blue))
    highlight3 = Highlighter((data, i, j) -> (j == 1), Crayon(foreground = :light_red, bold = true))
    pretty_table([ic.rnge ic.crit], header, header_crayon = crayon"yellow bold", formatters = (ft_printf("%5.0f", 1), ft_printf("%5.3f", 2:2)), highlighters = (highlight1, highlight2, highlight3))
end

function Base.show(io::IO, ic::Tuple{Vararg{InformationCriterion, N}}) where {N}
    header = ["# of factor"  string.(ic)...]
    highlights = map(x->Highlighter((data, i, j) -> data[i,j] == minimum(data[:,x]) && j > 1, Crayon(background = :light_blue, foreground = :white, bold = true)), 2:length(ic)+1)
    tbl = [first(ic).rnge mapreduce(x->x.crit, hcat, ic)]
    pretty_table(tbl, header, header_crayon = crayon"yellow bold", formatters = (ft_printf("%5.0f", 1), ft_printf("%5.3f", 2:length(ic)+1)), highlighters = (highlights...,))
end

function penalty(s::Type{P}, T, N) where P <: Union{IC1, PCp1}
    NtT = N*T
    NpT = N+T
    p1 = NpT/NtT
    p2 = log(NtT/NpT)
    p1*p2
end

function penalty(s::Type{P}, T, N) where P <: Union{IC2, PCp2}
    C2  = min(T, N)
    NtT = N*T
    NpT = N+T
    p1 = NpT/NtT
    p2 = log(C2)
    p1*p2
end

function penalty(s::Type{P}, T, N) where P <: Union{IC3, PCp3}
    C2  = min(T, N)
    log(C2)/C2
end

penalty(s::Type{S}, T, N, k) where S = penalty(s, T, N)

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

export FactorModel, subview, waldtest, describe, PValue, waldstat, Criteria,
       IC1, IC2, IC3, PCp1, PCp2, PCp3,
       AIC1, AIC2, AIC3, BIC1, BIC2, BIC3

end # module"
