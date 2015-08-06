@doc """
## Description
Object returned from `gausspr(::Array, ::Array)`.
## Members
* `alpha::Array`
* `kernel::FUnciton`: Kernel/covariance function
* `xmatrix::Array`: Designed matrix (scaled if scaled required in `gausspr`)
* `xcenter::Array`: A vector of centers (means) of design matrix. `[]` if no scaled.
* `xscale::Array`: A vector of scales (standard deviations) of design matrix. `[]` if no scaled.
* `ycenter::Array`: A vector of centers (means) of response matrix. `[]` if no scaled.
* `yscale::Array`: A vector of scales (standard deviations) of response matrix. `[]` if no scaled.
* `family::String`: A string of distribution
* `xlev`: A tuple of tuples of levels of factors in variable DataFrame
* `ylev`: A tuple of levels of response if response is categorical
* `kpar`: Keyword kernel parameters
""" ->
type GaussianProcessFittedMatrix <: GaussianProcessFitted
    alpha::Array
    kernel::Kernel
    xmatrix::Array
    xscale::Union(Nothing, Standardization)
    yscale::Union(Nothing, Standardization)
    family::Distribution
    xlev::Tuple
    ylev::Tuple
end

@doc """
## Description
Object returned from `gausspr(::Formula, ::DataFrame)`.
## Members
* `alpha::Array`
* `kernel::FUnciton`: Kernel/covariance function
* `xmatrix::Array`: Designed matrix (scaled if scaled required in `gausspr`)
* `formula::Formula`: Model formula 
* `xcenter::Array`: A vector of centers (means) of design matrix. `[]` if no scaled.
* `xscale::Array`: A vector of scales (standard deviations) of design matrix. `[]` if no scaled.
* `ycenter::Array`: A vector of centers (means) of response matrix. `[]` if no scaled.
* `yscale::Array`: A vector of scales (standard deviations) of response matrix. `[]` if no scaled.
* `family::String`: A string of distribution
* `xlev`: A tuple of tuples of levels of factors in variable DataFrame
* `ylev`: A tuple of levels of response if response is categorical
* `kpar`: Keyword kernel parameters
""" ->
type GaussianProcessFittedFormula <: GaussianProcessFitted
    alpha::Array
    kernel::Kernel
    xmatrix::Array
    formula::Formula
    xscale::Union(Nothing, Standardization)
    yscale::Union(Nothing, Standardization)
    family::Distribution
    xlev::Tuple
    ylev::Tuple
end

@doc """
## Description
It works for classification, regression, cox regression and poission regression The first/main purpose is to implement Gaussian process into Cox's model. Package glmnet is used to solve a general Ridge-regularized regression solver. 

## Arguments
* `formula::Formula`: Formula to express the model
* `data::DataFrame`: DataFrame of data to train
* `args`: Keyword arguments for the `gausspr(x::Array, y::Array)`

## Returns
A GaussianProcessfittedFormula object. 

## Examples
```julia
using DataFrames, GaussianProcess

dat = DataFrame(
    A = [0.4, 0.1, 0.7, 0.2],
    B = ["b", "a", "b", "a"],
    C = [0.3, 0.8, 0.1, 0.6],
    D = ["Yes", "No", "Yes", "No"]
    )

# use Formula-DataFrame input (classification)
gp = gausspr(D ~ A + B + C, dat, family = "binomial")

# predict class probabilities
pred = predict(gp, dat)
pred_class = predict(gp, dat, outtype = "response")
```
""" ->
function gausspr(formula::Formula, data::DataFrame; args...)
    x, y, xlev, ylev = modelmatrix(formula, data)
    o = gausspr(x, y; args...)
    return GaussianProcessFittedFormula(o.alpha, o.kernel, o.xmatrix, formula, o.xscale, o.yscale, o.family, xlev, ylev)
end


@doc """
## Description
It works for classification, regression, cox regression and poission regression The first/main purpose is to implement Gaussian process into Cox's model. Package glmnet is used to solve a general Ridge-regularized regression solver. 

## Arguments
* `x::Array`: Model design matrix
* `y::Array`: Response vector(regresion, binomial) or matrix (binomial, multinomial, survival)
* `ylevl = ()`: Tuple of response levels for classification
* `kernel = kernRBF`: Kernel / covariance function that returns a covariance matrix
* `var = 1.0`: Variance of Gaussian noise for regression only
* `scaled = true`: If to standardize design matrix `x` (and `y` if regression)
* `family = "gaussian"`: Distribution of y. Options: 'gaussian', 'binomial' (possibly 'poisson', 'multinomial', 'cox' in the future)
* `kpar`: Keyword arguments for `kernel` function

## Returns
A GaussianProcessfittedMatrix object. 

## Examples

```julia
using DataFrames, GaussianProcess

dat = DataFrame(
    A = [0.4, 0.1, 0.7, 0.2],
    B = ["b", "a", "b", "a"],
    C = [0.3, 0.8, 0.1, 0.6],
    D = ["Yes", "No", "Yes", "No"]
    )

x, y = modelmatrix(A ~ B + C + D, dat)
gp = gausspr(x, y, family = "gaussian");

# prediction
pred = predict(gp, x)
```
""" ->
function gausspr(x::Array, y::Array; family = Normal(0, 1.0), ylev = (), kernel = kernelRBF(5), scaled = true)
    xscale = yscale = nothing
    if scaled
        x, xscale = standardize(x, rm_const = true)
        if isa(family, Normal)
            y, yscale = standardize(y, rm_const = false)
        end
    end
    K = kernelMatrix(kernel, x)
    # cholesky decomposition K = R'R,  f = theta = K alpha = R' beta =>  R alpha = beta, F=R'
    F = try
            chol(K)
        catch
            chol(K + 1.e-8*eye(size(K,1)))
        end
    #
    if isa(family, Normal)
        lambda = std(family)^2 / size(x,1) 
    elseif isa(family, Binomial)
        lambda = 1.0 / size(x,1)
    end
    mod = glmnet(F.', y, family, lambda = [lambda], alpha = 0.0, intercept = false, standardize = false);
    alpha = F \ mod.betas.ca;
    return GaussianProcessFittedMatrix(alpha, kernel, x, xscale, yscale, family, (), ())
end


# predict function
@doc """
## Description
Prediction on new dataset

## Arguments
* `gp::GaussianProcessfittedMatrix`: Object from `gausspr(::Array, ::Array)`
* `newdata::Array`: Design matrix of data to predict
* `outtype = "prob"`: Keyword argument either "prob" or "class"

## Returns
Predicted probabilities, classes or response depended on input model
""" ->
function predict(gp::GaussianProcessFittedMatrix, newdata::Array; outtype = "prob")
    if gp.xscale == []
        x = newdata
    else 
        x, = standardize(newdata, gp.xcenter, gp.xscale, rmconst = true)
    end
    #
    x = gp.kernel(x, gp.xmatrix; gp.kpar...)
    pred = x*gp.alpha;
    if gp.family == "binomial"
        pred = 1 ./ (1 + exp(-pred))
        if outtype == "prob"
            pred = [1. - pred pred]
        elseif gp.ylev != ()
            pred = ifelse(pred .< 0.5, ylev[1], ylev[2])
        end
    elseif gp.family == "gaussian"
        if gp.ycenter == []
            pred = pred .* gp.yscale .+ gp.ycenter
        end
    end
    return pred
end


@doc """
# Description
Prediction on new dataset
# Arguments
* `gp::GaussianProcessfittedFormula`: Object from `gausspr(::Formula, ::DataFrame)`
* `newdata::DataFrame`: DataFrame of data to predict
* `outtype = "prob"`: Keyword argument either "prob" or "class"
# Returns
Predicted probabilities, classes or response depended on input model
""" ->
function predict(gp::GaussianProcessFittedFormula, newdata::DataFrame; outtype = "prob")
    formula = deepcopy(gp.formula)
    formula.lhs = nothing
    x, = modelmatrix(formula, newdata, xlev = gp.xlev, ylev = gp.ylev)

    if gp.xscale != []
        x, = standardize(x, gp.xcenter, gp.xscale, rmconst = true)
    end
    #
    x = gp.kernel(x, gp.xmatrix; gp.kpar...)
    pred = x*gp.alpha;
    if gp.family == "binomial"
        pred = 1 ./ (1 + exp(-pred))
        if outtype == "prob"
            pred = [1. - pred pred]
        else 
            pred = ifelse(pred .< 0.5, gp.ylev[1], gp.ylev[2])
        end
    elseif gp.family == "gaussian"
        if gp.ycenter == []
            pred = pred .* gp.yscale .+ gp.ycenter
        end
    end
    return pred
end
