# Gaussian Process

Gaussian process classification, regression, and Cox regression for time-to-event data
This package has not yet registered in the METADATA. To install, run the following command.

```julia
# Install
Pkg.clone("git://github.com/linxihui/GaussianProcess.jl")

```

Since it depends on the newest [`GLMNet`](https://github.com/linxihui/GLMNet.jl), which is in my branch and has not been merge to the main branch, one has to clone and build the package as

```julia
Pkg.clone("git://github.com/linxihui/GLMNet.jl")
Pkg.build("GLMNet")
```

# Syntax

The main function of this package is `gausspr`, which have two methods like R

- `gausspr(::Formula, ::DataFrame, ::Distribution)`
- `gausspr(::Array, ::Array, ::Distribution)`

In formula, symbol `..` is supported, which means all other variables in `DataFrame`, equivalent to
the `.` symbol in R formula. For example `y ~ ..` means `y` is the response and all other variable except `y`
in the `DataFrame` are predictors.  

For `Distribution`, currently 5 different types are supported. 
- `Normal()` : regression of continuous response 
- `Binomial()` and `Multinomial()`: classifications
- `Poisson()`:  count data
- `CoxPH()`: survival or time-to-event data

Different Kernels are available, including 
- `kernelLinear()` or `kernelVanilla`
- `kernelPoly(degree=3, offset=1.0)`
- `kernelRBF(σ=1.0)`: `σ` is the standard deviation
- `kernelLaplace(σ=1.0)`: `σ` is the scale parameter
- `kernelMatern(ν=2.0, σ=1.0)`
- `kernelUser(K)`: `K` is a user defined kernel matrix


# A Quick Guide

```julia
julia> using DataFrames, GaussianProcess

julia> dat = DataFrame(
           A = [0.4, 0.1, 0.7, 0.2],
           B = ["b", "a", "b", "a"],
           C = [0.3, 0.8, 0.1, 0.6],
           D = ["Yes", "No", "Yes", "No"]
           );

julia> gp = gausspr(D ~ A + B + C, dat, Binomial());

# predict class probabilities
julia> pred = predict(gp, dat, outtype = :prob)
4x2 Array{Float64,2}:
 0.432865  0.567135
 0.597388  0.402612
 0.394668  0.605332
 0.575207  0.424793

julia> pred_class = predict(gp, dat, outtype = :class)
4-element Array{String,1}:
 "Yes"
 "No"
 "Yes"
 "No"

# use x-y input (regression)
julia> x, y = modelmatrix(A ~ B + C + D, dat)
(
4x5 Array{Float64,2}:
 0.0  1.0  0.3  0.0  1.0
 1.0  0.0  0.8  1.0  0.0
 0.0  1.0  0.1  0.0  1.0
 1.0  0.0  0.6  1.0  0.0,

[0.4,0.1,0.7,0.2],(("a","b"),(),("No","Yes")),())


julia> gp2 = gausspr(x, y, Normal(0, 1), kernel = kernelPoly(3, 1));

julia> pred2 = predict(gp2, x)
4-element Array{Float64,1}:
 0.409322
 0.0996975
 0.690604
 0.200377 
```

# Binary Classification

```julia
julia> using RDatasets

julia> srand(123)

julia> kyphosis = dataset("rpart", "kyphosis");

julia> kyphosis[:Kyphosis] = convert(Array, kyphosis[:Kyphosis]);

julia> iTrain = sample(1:size(kyphosis, 1), 40, replace = false);

julia> iTest = setdiff(1:size(kyphosis, 1), iTrain);

julia> ky_mod = gausspr(Kyphosis ~ .., kyphosis[iTrain, :], Binomial());

julia> ky_pred = predict(ky_mod, kyphosis[iTest, :], outtype = :prob);

julia> out = ROC(kyphosis[iTest, :Kyphosis].data, ky_pred[:, 2]);

julia> out.auc
0.8715277777777777

julia> plot(out)
```
![](https://rawgit.com/linxihui/Misc/master/Images/GaussianProcess.jl/ROC.svg)


# Regression

```julia
julia> srand(456)

julia> boston = dataset("MASS", "Boston");

julia> iTrain = sample(1:size(boston, 1), 300, replace = false);

julia> iTest = setdiff(1:size(boston, 1), iTrain);

julia> mod_boston = gausspr(MedV ~ .., boston[iTrain, :], kernel = kernelMatern(2, 1));

julia> pred_boston = predict(mod_boston, boston[iTest, :]);

julia> err_boston = sqrt(mean((convert(Array, boston[iTest, :MedV]) - pred_boston).^2))
3.3446708896894983
```
