using Factotum, Statistics, LinearAlgebra, Test
T, n, r = (200, 10, 6)
x = rand(T,n)                  # generate data
fm = Factotum.FactorModel(x, 6; scale = true)      # fit factor model

F = Factotum.factors(fm)
Λ = Factotum.loadings(fm)
σ = Factotum.sdev(fm)
## Check diagonality cov(F)

Σ = cov(F; corrected = false);

@test Σ ≈ diagm(0=>σ.^2)
@test Λ'Λ/n ≈ diagm(0=>ones(r))


## Replicate Ng and Bai (2002)
function simulate_factormodel(r, T, N)
    F = randn(T,r)
    Λ = randn(r,N)
    e = sqrt(r).*randn(T,N)
    F*Λ .+ e 
end

X = simulate_factormodel(1, 251, 78)
fm = Factotum.FactorModel(X, 10; scale = true)
Factotum.informationcriteria((IC1, IC2), fm, 10)

X = simulate_factormodel(3, 100, 20)
fm = Factotum.FactorModel(X, 10; scale = true)
ic = Factotum.informationcriteria((IC1, IC2), fm, 10)

findmin(ic)
## Test the proportion of the variance explained

