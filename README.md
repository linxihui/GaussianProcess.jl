# Gaussian Process

Gaussian process classification, regression, and Cox regression for time-to-event data (not yet implemented). 
This package has not yet registered in the METADATA. To install, run the following command.

```julia
# Install

Pkg.clone("git://github.com/linxihui/GaussianProcess.jl")

```

# Quick walk-through

```julia
using DataFrames, GaussianProcess

dat = DataFrame(
	A = [0.4, 0.1, 0.7, 0.2],
	B = ["b", "a", "b", "a"],
	C = [0.3, 0.8, 0.1, 0.6],
	D = ["Yes", "No", "Yes", "No"]
	)

# use Formula-DataFrame input (classification)
gp = gausspr(D ~ A + B + C, dat, family = "binomial")

# predict class probabilities
pred = predict(gp, dat)
pred_class = predict(gp, dat, outtype = "response")

# use x-y input (regression)
x, y = modelmatrix(A ~ B + C + D, dat)
gp2 = gausspr(x, y, family = "gaussian");

# prediction
pred2 = predict(gp2, x)
```
