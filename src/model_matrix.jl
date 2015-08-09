@doc """
## Description
Create a model matrix from Formual and DataFrame input. Note that all factors (categorical variables) are represented as FULL dummy variables

## Arguments
* `formula::Formual`: A `Formula` object from `DataFrames`. `..` can be used to indicate all variables else in `data` (other than those have been mentioned), which is whats the same as `.` in R.
* `data::DataFrame`: A data frame of data.
* `xlev = ()`: A tuple of tuple of levels of categorical independent variables Usually not specified.
* `ylev = ()`: A tuple of levels of the categorical response. Usually not specified.

## Returns
* A tuple of desig matrix, response matrix, 
""" ->
function modelmatrix(formula::Formula, data::DataFrame; xlev = (), ylev = ())
    if formula.rhs == :.. || (isa(formula.rhs, Expr) && formula.rhs.args[length(formula.rhs.args)] == :..)
		formula = deepcopy(formula)
        args = isa(formula.lhs, Symbol)? [formula.lhs] : isa(formula.lhs, Nothing)? [] : formula.lhs.args
        if isa(formula.rhs, Symbol)
            dots = setdiff(names(data), args)
            formula.rhs = (1 == length(dots))? args[1] : Expr(:call, :+, dots...)
        else
            dots = setdiff(names(data), [args, formula.rhs.args])
            formula.rhs.args = [formula.rhs.args[1:(length(formula.rhs.args) - 1)], dots]
        end
    end
    mf = ModelFrame(formula, data).df
    # response
    if isa(formula.lhs, Nothing) 
        response = []
    else
        response = mf[:, 1].data;
        if ~(typeof(response).parameters[1] <: Number)
            if ylev == ()
                ylev = tuple(sort(levels(response))...)
            end
            response = convert(Matrix{Float64}, [i == j for i in response, j in ylev])
        end
        mf = mf[:, 2:size(mf, 2)]
    end
    # design matrix
    xlevempty = xlev == ()
    if typeof(mf[:, 1]).parameters[1] <: Number
        designmatrix = mf[:, 1].data
        lev = ()
    else 
        if xlevempty
            lev = tuple(sort(levels(mf[:, 1]))...)
        else
            lev = tuple(xlev[1]...)
        end
        designmatrix = [float64(i == j) for i in mf[:, 1], j in lev]
    end
    xlevnew = [lev]
    icol = 2;
    while icol <= size(mf, 2)
        if typeof(mf[:, icol]).parameters[1] <: Number
            designmatrix = [designmatrix mf[:, icol].data]
            lev = ()
        else 
            if xlevempty
                lev = tuple(sort(levels(mf[:, icol]))...)
            else
                lev = tuple(xlev[icol]...)
            end
            designmatrix = [designmatrix [float64(i == j) for i in mf[:, icol], j in lev]]
        end
        xlevnew = [xlevnew, lev]
        icol = icol + 1
    end
    xlev = tuple(xlevnew...)
    return convert(Matrix{Float64}, designmatrix), response, xlev, ylev
end
