# Gaussian Process

# A quick walk-through

```julia
using DataFrames, GaussianProcess

dat = DataFrame(
	A = rand(4),
	B = ["a", "b", "c", "b"],
	C = randn(4),
	D = ["A", "B", "A", "B"]
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
