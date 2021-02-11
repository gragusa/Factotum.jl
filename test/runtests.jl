using Factotum, Random, Statistics, LinearAlgebra, Test
T, n, r = (100, 10, 6)
Random.seed!(101020)
x = rand(T,n)                  # generate data
fm = Factotum.FactorModel(x, 6; scale = true)      # fit factor model

F = Factotum.factors(fm)
Λ = Factotum.loadings(fm)
σ = Factotum.sdev(fm)
## Check diagonality cov(F)

Σ = cov(F; corrected = false);

@test Σ ≈ diagm(0=>σ.^2)
@test Λ'Λ/n ≈ diagm(0=>ones(r))

## Test the proportion of the variance explained

## Test

####### Call R, start ################################################
R"
x <- $x
f <- princomp(x)
var <- summary(f)"
####### Call R, end ###################################################

f_r = reval("f")     # export R factor model in julia workspace
var_r = reval("var") # export R factor variance in julia workspace

@test isapprox(convert(Array,var_r[1]), f_julia.sdev)   # testing variances
@test isapprox(convert(Array,f_r[2]), f_julia.loadings) # testing loadings
@test isapprox(convert(Array,f_r[3]), f_julia.loadings) # testing loadings
@test isapprox(convert(Array,f_r[6]), f_julia.factors)  # testing factors