abstract kernel

# Linear kernel
immutable kernelVanilla <: kernel
end

typealias kernelLinear kernelVanilla

kernelMatrix(K::kernelVanilla, x::Array{Real}, y::Array{Real} = x) = x*y.'

# Polynomial kernel
# 	(x'y + c)^d, where c = offect, d = degree
immutable kernelPoly <: kernel
    degree::Int
    offset::Real
    function kernelPoly(degree::Int = 3, offset::Real = 1)
        degree > 0 || error("degree must be positive.")
        new(degree, offset)
    end
end

kernelMatrix(K::kernelPoly, x::Array{Real}, y::Array{Real} = x) = (x*y.' .+ K.offset).^K.degree

# RBF kernel
# 	exp(-d^2/(2*σ²)
immutable kernelRBF <: kernel
	σ::Real
	function kernelRBF(σ::Real = 1.0)
		σ > 0 || error("σ must be positive.")
		new(σ)
	end
end

function kernelMatrix(K::kernelRBF, x::Array{Real}, y::Array{Real} = x)
    exp(-rowwiseDist(x, y, squared = true) / (2*K.σ^2))
end

# Laplace kernel
# 	exp(-d/σ), where d is L2 distance
immutable kernelLaplace <: kernel
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
immutable kernelMatern <: kernel
	ν::Real
	σ::Real
	function kernelMatern(ν = 2.0, σ = 1.0)
		(ν > 0 && σ > 0) || error("ν and σ must be positive.")
		new(ν, σ)
	end
end

function kernelMatrix(K::kernelMatern, x::Array{Number}, y::Array{Number} = x)
    d = (sqrt(2)*K.ν / K.σ)*rowwiseDist(x, y) + 1e-20*eye(size(x, 1));
    besselk(K.ν, d) .* (d .^ K.ν) / (gamma(K.ν) * 2.0^(K.ν - 1))
end


# User define kernel (matrix)
immutable kernelUser <: kernel
	K::Matrix{Real}
	function kernelUser(K::Matrix{Real})
		isposdef(K) || error("Matrix must be positive definite.")
		new(K)
	end
end

kernelMatrix(K::kernelUser, x::Array{Number}, y::Array{Number} = x) = K.K
