using Factotum
using Base.Test
using RCall
# write your own tests here

x = rand(100,10)
f_julia = Factotum.FactorModel(x)

R"  
x <- $x
f <- princomp(x)
var <- summary(f)"

f_r = reval("f")
var_r = reval("var")

@test isapprox(convert(Array,var_r[1]), f_julia.sdev)
@test isapprox(convert(Array,f_r[6]), f_julia.factors)