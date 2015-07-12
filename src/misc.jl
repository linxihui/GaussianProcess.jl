# compute row-wise Euclidean distance
function rowwiseDist(x::Array; squared = false)
	xNorm = repmat(sum(x.^2, 2), 1, size(x, 1))
	dist2 = xNorm + xNorm' - 2*x*x'
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
	if squared
		return dist2
	else 
		return sqrt(dist2)
	end
end

# standization
function standardize(x::Array, center::Array, scale::Array; rmconst = false)
	n = size(x, 1)
	xscaled = (x - repmat(center, n)) ./ repmat(scale, n)
	if rmconst
		xscaled = xscaled[:, convert(Array{Bool, 1}, [sd > 0 for sd in scale])]
	end
	return xscaled, center, scale, rmconst
end

function standardize(x::Array; rmconst = false)
	center = mean(x, 1)
	scale = std(x, 1)
	return standardize(x, center, scale, rmconst = rmconst)
end