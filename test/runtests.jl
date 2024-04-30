using Test
using SparseGaussianProcess

# Test SGP constructor
sgp = SparseGaussianProcess.SGP()

# Get the shape of the inducing inputs
println(sgp.Z)

# Test set_inducing_inputs!
@testset "set_inducing_inputs!" begin
    X = rand(10,3)  # Example training data
    SparseGaussianProcess.set_inducing_inputs!(sgp, X)

    @test !isnothing(sgp.Z)
    @test size(sgp.Z, 2) == size(X, 2)
    @test sgp.nz == sgp.options["n_inducing"]
end
