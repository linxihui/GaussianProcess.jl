type GaussianProcessFittedMatrix <: GaussianProcessFitted
	alpha::Array
	kernel::Function
	xmatrix::Array
	xcenter::Array
	xscale::Array
	family::String
	xlev
	ylev
	kpar
end

type GaussianProcessFittedFormula <: GaussianProcessFitted
	alpha::Array
	kernel::Function
	xmatrix::Array
	formula::Formula
	xcenter::Array
	xscale::Array
	family::String
	xlev
	ylev
	kpar
end


function gausspr(formula::Formula, data::DataFrame; args...)
	x, y, xlev, ylev = modelmatrix(formula, data)
	o = gausspr(x, y; args...)
	return GaussianProcessFittedFormula(o.alpha, o.kernel, o.xmatrix, formula, o.xcenter, o.xscale, o.family, xlev, ylev, o.kpar)
end


function gausspr(x::Array, y::Array;  ylev = (), kernel = kernRBF, var = 1.0, scaled = true, family = "gaussian", kpar...)
	x_center = x_scale = y_center = y_scale = []
	if scaled
		x, x_center, x_scale, = standardize(x, rmconst = true)
	end
	K = kernel(x; kpar...)
	# cholesky decomposition K = R'R,  f = theta = K alpha = R' beta =>  R alpha = beta, F=R'
	F = try
			chol(K)
		catch
			chol(K + 1.e-8*eye(size(K,1)))
		end
	#
	if family == "gaussian"
		lambda = var / size(x,1) 
		fmly = Normal()
	elseif family == "binomial"
		lambda = 1.0 / size(x,1)
		fmly = Binomial()
	end
	mod = glmnet(F.', y, fmly, lambda = [lambda], alpha = 0.0, intercept = false, standardize = false);
	alpha = F \ mod.betas.ca;
	return GaussianProcessFittedMatrix(alpha, kernel, x, x_center, x_scale, family, (), ylev, kpar)
end


# predict function

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
		if outtype == "prob"
			pred = 1 ./ (1 + exp(-pred))
			pred = [1. - pred pred]
		elseif gp.ylev != ()
			pred = ifelse(pred .< 0.5, ylev[1], ylev[2])
		end
	end
	return pred
end


function predict(gp::GaussianProcessFittedFormula, newdata::DataFrame; outtype = "prob")
	formula = gp.formula
	formula.lhs = nothing
	x, = modelmatrix(formula, newdata, xlev = gp.xlev, ylev = gp.ylev)
	if gp.xscale != []
		x, = standardize(x, gp.xcenter, gp.xscale, rmconst = true)
	end
	#
	x = gp.kernel(x, gp.xmatrix; gp.kpar...)
	pred = x*gp.alpha;
	if gp.family == "binomial"
		if outtype == "prob"
			pred = 1 ./ (1 + exp(-pred))
			pred = [1. - pred pred]
		else 
			pred = ifelse(pred .< 0.5, gp.ylev[1], gp.ylev[2])
		end
	end
	return pred
end
