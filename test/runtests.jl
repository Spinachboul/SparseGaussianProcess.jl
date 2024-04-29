using Test
using SparseGaussianProcess  # Import your package module

# Test set_inducing_inputs!
sgp = SparseGaussianProcess.SGP()
X = rand(10, 2)  # Example training data
SparseGaussianProcess.set_inducing_inputs!(sgp, X)
@test !isnothing(sgp.Z)
@test size(sgp.Z, 2) == size(X, 2)
@test sgp.nz == sgp.options["n_inducing"]  # Check if number of inducing inputs is correct