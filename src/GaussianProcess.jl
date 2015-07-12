module GaussianProcess
	using DataFrames, Distributions, GLMNet

	export 
		gausspr, # Gaussian process classification, regression
		predict, # prediction function for Gaussian process model
		kernVanilla, # Vanilla / linear kernel
		kernPoly, # polynomial kernel
		kernRBF, # radial based / Gaussian kernel
		kernLaplace, # Laplace kernel
		kernMatern, # Matern class kernel
		rowwiseDist, # calculate row-wise Euclidean distance
		standardize, # standardize matrices
		modelmatrix  # compute design matrix, response from Formula-DataFrame inputs

	abstract GaussianProcessFitted # output of Gaussian process model

	include("misc.jl")
	include("kernels.jl")
	include("model_matrix.jl")
	include("gaussian_process.jl")

end
