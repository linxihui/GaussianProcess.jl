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
    center::Vector{Real}
    scale::Vector{Real}
    zero_scale_column::Vector{Bool}
    function Standardization(center::Vector, scale::Vector = repmat([1], length(scale)), zero_scale_column::Vector = repmat([false], length(scale)))
        length(center) == length(scale) == length(zero_scale_column) || error("Length of center and scale must match row number of x.")
        new(center, scale, zero_scale_column)
    end
end

function transform(preproc::Standardization, data::Array)
    n = size(data, 1);
    (data[:, !preproc.zero_scale_column] - repmat(preproc.center.', n)) ./ repmat(preproc.scale.', n)
end

function inverseTransform(preproc::Standardization, data::Array)
    n = size(data, 1);
    data .* repmat(preproc.scale.', n) + repmat(preproc.center.', n) 
end
    
# standization
function standardize(x::Union(Vector, Matrix), center::Vector = mean(x,1)[:], scale::Vector = std(x, 1)[:]; rm_const = false)
    size(x, 2) == length(center) == length(scale) || error("Length of center and scale must match row number of x.")
	n = size(x, 1)
    zero_scale_col = repmat([false], size(x,2))
    if rm_const 
        zero_scale_col = (scale .== 0)
        if any(zero_scale_col)
            x = x[:, zero_scale_col]
            center = center[zero_scale_col]
            scale = scale[zero_scale_col]
        end
    end
	xscaled = (x - repmat(center.', n)) ./ repmat(scale.', n)
	return xscaled, Standardization(center, scale, zero_scale_col)
end
