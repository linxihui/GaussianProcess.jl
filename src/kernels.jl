abstract Kernel

# Linear Kernel
immutable kernelVanilla <: Kernel
end

typealias kernelLinear kernelVanilla

kernelMatrix(K::kernelVanilla, x::Array, y::Array = x) = x*y.'

# Polynomial kernel
# 	(x'y + c)^d, where c = offect, d = degree
immutable kernelPoly <: Kernel
    degree::Int
    offset::Float64
    function kernelPoly(degree::Int = 3, offset::Real = 1)
        degree > 0 || error("degree must be positive.")
        new(degree, offset)
    end
end

kernelMatrix(K::kernelPoly, x::Array, y::Array = x) = (x*y.' .+ K.offset).^K.degree

# RBF kernel
# 	exp(-d^2/(2*σ²)
immutable kernelRBF <: Kernel
	σ::Float64
	function kernelRBF(σ::Real = 1.0)
		σ > 0 || error("σ must be positive.")
		new(σ)
	end
end

function kernelMatrix(K::kernelRBF, x::Array, y::Array = x)
    exp(-rowwiseDist(x, y, squared = true) / (2*K.σ^2))
end

# Laplace kernel
# 	exp(-d/σ), where d is L2 distance
immutable kernelLaplace <: Kernel
	σ::Real
	function kernelLaplace(σ::Real = 1.0)
		σ > 0 || error("σ must be positive.")
		new(σ)
	end
end

kernelMatrix(K::kernelLaplace, x::Array, y::Array = x) = exp(-rowwiseDist(x, y)) / σ

# Matern class / Bessel kernel
# 	@reference https://en.wikipedia.org/wiki/Gaussian_process#Usual_covariance_functions
# 	where l => d, ρ => σ
immutable kernelMatern <: Kernel
	ν::Real
	σ::Real
	function kernelMatern(ν = 2.0, σ = 1.0)
		(ν > 0 && σ > 0) || error("ν and σ must be positive.")
		new(ν, σ)
	end
end

function kernelMatrix(K::kernelMatern, x::Array, y::Array = x)
    d = (sqrt(2)*K.ν / K.σ)*rowwiseDist(x, y) + 1e-20*eye(size(x, 1));
    besselk(K.ν, d) .* (d .^ K.ν) / (gamma(K.ν) * 2.0^(K.ν - 1))
end


# User define kernel (matrix)
immutable kernelUser <: Kernel
	K::Matrix
	function kernelUser(K::Matrix)
		isposdef(K) || error("Matrix must be positive definite.")
		new(K)
	end
end

kernelMatrix(K::kernelUser, x::Array, y::Array = x) = K.K
