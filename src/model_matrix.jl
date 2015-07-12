function modelmatrix(formula::Formula, data::DataFrame; xlev = (), ylev = ())
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
