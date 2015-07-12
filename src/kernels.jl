# Vanilla / Linear kernel
function kernVanilla(x::Array)
	return x*x.'
end

function kernVanilla(x::Array, y::Array)
	return x*y.'
end


# Polynomial kernel
# 	(x'y + c)^d, where c = offect, d = degree
function kernPoly(x::Array, degree = 3, offset = 1.0)
	return (x*x.' + offset).^degree
end

function kernPoly(x::Array, y::Array; degree = 3, offset = 1.0)
	return (x*y.' .+ offset).^degree
end


# RBF kernel
# 	exp(-d^2/(2*sigma^2)
function kernRBF(x::Array; sigma= 1.0)
	return exp(-rowwiseDist(x, squared = true) / (2*sigma^2))
end

function kernRBF(x::Array, y::Array; sigma= 1)
	return exp(-rowwiseDist(x, y, squared = true) / (2*sigma^2))
end


# Laplace kernel
# 	exp(-d / sigma), where d is L2 distance
function kernLaplace(x::Array; sigma = 1.0)
	return exp(-rowwiseDist(x) / sigma)
end

function kernLaplace(x::Array, y::Array; sigma = 1.0)
	return exp(-rowwiseDist(x, y) / sigma)
end


# Matern class / Bessel kernel
# 	@reference https://en.wikipedia.org/wiki/Gaussian_process#Usual_covariance_functions
# 	where l == d, rho == sigma
function kernMatern(x::Array; nu = 2.0, sigma = 1.0) 
	dd = (sqrt(2)*nu / sigma)*rowwiseDist(x) + 1e-20*eye(size(x, 1))
	return besselk(nu, dd) .* (dd .^ nu) / (gamma(nu) * 2.0^(nu - 1))
end

function kernMatern(x::Array, y::Array; nu = 2.0, sigma = 1.0) 
	dd = (sqrt(2)*nu / sigma)*rowwiseDist(x, y) + 1e-20*eye(size(x, 1));
	return besselk(nu, dd) .* (dd .^ nu) / (gamma(nu) * 2.0^(nu - 1))
end
