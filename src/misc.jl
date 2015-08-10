# compute row-wise Euclidean distance
function rowwiseDist(x::Array; squared = false)
    xNorm = repmat(sum(x.^2, 2), 1, size(x, 1))
    dist2 = xNorm + xNorm' - 2*x*x'
    if any(dist2 .< 0) dist2 = abs(dist2) end
    if squared
        return dist2
    else
        return sqrt(dist2)
    end
end

function rowwiseDist(x::Array, y::Array; squared = false)
    xNorm = repmat(sum(x.^2, 2), 1, size(y, 1))
    yNorm = repmat(sum(y.^2, 2), 1, size(x, 1))
    dist2 = xNorm + yNorm' - 2*x*y'
    if any(dist2 .< 0) dist2 = abs(dist2) end
    if squared
        return dist2
    else
        return sqrt(dist2)
    end
end

abstract PreProcess

immutable Standardization <: PreProcess
    center::Vector{Float64}
    scale::Vector{Float64}
    zero_scale_column::Vector{Bool}
    function Standardization(center::Vector, scale::Vector = repmat([1], length(scale)), zero_scale_column::Vector = repmat([false], length(scale)))
        length(center) == length(scale) == length(zero_scale_column) || error("Length of center and scale must match row number of x.")
        new(center, scale, zero_scale_column)
    end
end

function transform(preproc::Standardization, data::Array)
    n = size(data, 1)
    out = (data[:, !preproc.zero_scale_column] - repmat(preproc.center.', n)) ./ repmat(preproc.scale.', n)
    if size(data, 2) == 1 out[:] else out end
end

function inverseTransform(preproc::Standardization, data::Array)
    n = size(data, 1)
    out = data .* repmat(preproc.scale.', n) + repmat(preproc.center.', n)
    if size(data, 2) == 1 out[:] else out end
end
    
# standization
function standardize(x::Union(Vector, Matrix), center::Vector = mean(x,1)[:], scale::Vector = std(x, 1)[:]; rm_const = false)
    size(x, 2) == length(center) == length(scale) || error("Length of center and scale must match row number of x.")
    n = size(x, 1)
    zero_scale_col = repmat([false], size(x,2))
    if rm_const 
        zero_scale_col = convert(Vector{Bool}, (scale .== 0))
        if any(zero_scale_col)
            x = x[:, zero_scale_col]
            center = center[zero_scale_col]
            scale = scale[zero_scale_col]
        end
    end
    xscaled = (x - repmat(center.', n)) ./ repmat(scale.', n)
    if isa(x, Vector) xscaled = xscaled[:] end
    return xscaled, Standardization(center, scale, zero_scale_col)
end


# transform links to probabilities, classes, risks
function transform_link(link::Array, outtype::Symbol, ylev::Tuple)
    if outtype == :link
        return link
    else
        pred = exp(link)
        pred = pred ./ repmat(sum(pred, 2), 1, size(pred, 2))
        if outtype == :prob
            return pred
        elseif outtype == :class
            if ylev != ()
                return convert(Vector{String}, [ylev[indmax(pred[i,:])] for i = 1:size(pred, 1)])
            else
                error("gp.ylev is not defined.")
            end
        else
            error("For classification, outtype must be :link, :prob, or :class.")
        end
    end
end

function transform_link(family::Union(Poisson, CoxPH), link::Array, outtype::Symbol)
    if outtype == :link
        return link
    elseif outtype == :risk
        return exp(link)
    else
        if isa(family, Poisson)
            error("For Poisson model, outtype must be :link or :risk.")
        else
            error("For Cox model, outtype must be :link or :risk.")
        end
    end
end


type ROC
    fp::Vector{Float64}
    tp::Vector{Float64}
    threshold::Vector{Float64}
    auc::Float64
end

function ROC(obs::Vector, prob::Vector)
    lev = sort(unique(obs))
    length(lev) == 2 || error("obs must have exactly two classes.")
    length(obs) == length(prob) || error("obs and prob must have same length.")
    obs = [0,1][1 + int(obs .== lev[2])]
    uniq_prob = sort(unique(prob), rev = true)
    threshold = (uniq_prob[1:(length(uniq_prob)-1)] + uniq_prob[2:length(uniq_prob)]) / 2.0
    threshold = [Inf, threshold..., -Inf]
    dist_obs = counts(obs, 0:1)
    function compute_fptp(obs::Vector, prob::Vector, thresh::Real)
        pred = int(prob .> thresh)
        tab = counts(obs, pred, (0:1, 0:1))
        (tab[:, 2] ./ dist_obs).'
    end
    n = length(threshold)
    fp_tp = [[compute_fptp(obs, prob, thresh) for thresh in threshold]...]
    auc = sum((fp_tp[2:n, 2] + fp_tp[1:(n-1), 2]) .* diff(fp_tp[:, 1])) / 2.0
    ROC(fp_tp[:, 1], fp_tp[:, 2], threshold, auc)
end

import Gadfly.plot

function plot(object::ROC, args...; kargs...)
    plot(x = object.fp, y = object.tp, Gadfly.Geom.line(preserve_order = true),
        Gadfly.Guide.xlabel("False Positive"), Gadfly.Guide.ylabel("True Positive"),
        args...; kargs...)
end
