using Base.Test, DataFrames, GaussianProcess

dat = DataFrame(
	A = [0.4, 0.1, 0.7, 0.2],
	B = ["b", "a", "b", "a"],
	C = [0.3, 0.8, 0.1, 0.6],
	D = ["Yes", "No", "Yes", "No"]
	);

gp = gausspr(D ~ A + B + C, dat, family = "binomial")

# predict class probabilities
pred = predict(gp, dat)
pred_class = predict(gp, dat, outtype = "response")

# use x-y input (regression)
x, y = modelmatrix(A ~ B + C + D, dat)

gp2 = gausspr(x, y, family = "gaussian", kernel = kernPoly, degree = 3, offset = 1);
pred2 = predict(gp2, x)

gp_alpha = [0.3800255, -0.3519039, 0.3707387, -0.3586464]
gp_pred_ground_true = [ 
	0.3800255 0.6199745;
	0.6480962 0.3519038;
	0.3707387 0.6292613;
	0.6413536 0.3586464 ]

x_ground_true = [
	0.0  1.0  0.3  0.0  1.0;
	1.0  0.0  0.8  1.0  0.0;
	0.0  1.0  0.1  0.0  1.0;
	1.0  0.0  0.6  1.0  0.0 ]

@test gp.ylev == ("No","Yes")
@test_approx_eq_eps gp_alpha gp.alpha 1e-2
@test_approx_eq_eps pred gp_pred_ground_true 1e-2
@test_approx_eq x x_ground_true
@test_approx_eq gp2.xmatrix standardize(x)[1]
