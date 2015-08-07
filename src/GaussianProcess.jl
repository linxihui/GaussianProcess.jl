module GaussianProcess
	using Docile, DataFrames, Distributions, GLMNet

	import DataFrames.predict
	export 
        GaussianProcessFitted,
        GaussianProcessFittedMatrix,
        GaussianProcessFittedFormula,
		gausspr, # Gaussian process classification, regression
		predict, # prediction function for Gaussian process model
        Kernel, # abstract kernel type
		kernelVanilla, # Vanilla / linear kernel
		kernelPoly, # polynomial kernel
		kernelRBF, # radial based / Gaussian kernel
		kernelLaplace, # Laplace kernel
		kernelMatern, # Matern class kernel
		rowwiseDist, # calculate row-wise Euclidean distance
        Standardization, 
		standardize, # standardize matrices
        transform,  # apply Standardization object to new data
        inverseTransform, # transform standardized data to original scale
		modelmatrix  # compute design matrix, response from Formula-DataFrame inputs

	abstract GaussianProcessFitted # output of Gaussian process model

	include("misc.jl")
	include("kernels.jl")
	include("model_matrix.jl")
	include("gaussian_process.jl")

end
