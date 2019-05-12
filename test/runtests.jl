using Factotum, Base.Test, RCall

x = rand(100,10)                  # generate data
f_julia = Factotum.FactorModel(x) # fit factor model

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