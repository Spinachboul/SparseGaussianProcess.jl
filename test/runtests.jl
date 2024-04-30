using Test
using SparseGaussianProcess
using BenchmarkTools


# Function to benchmark the training of the SparseGaussianProcess model
 function benchmark_training(n_samples::Int)
    # Create a SparseGaussianProcess model
    sgp = SparseGaussianProcess.SGP()
    y = rand(100,1)
    sgp.training_points["Y"] = y
    sgp.options["use_hetero_noise"] = false

    # Set the training data
    SparseGaussianProcess._new_train!(sgp)

end

# Test the benchmark
benchmark_training(1000)


# Test set_inducing_inputs!
@testset "set_inducing_inputs!" begin
    sgp = SparseGaussianProcess.SGP()
    X = rand(10,3)  # Example training data
    SparseGaussianProcess.set_inducing_inputs!(sgp, X)

    @test !isnothing(sgp.Z)
    @test size(sgp.Z, 2) == size(X, 2)
    @test sgp.nz == sgp.options["n_inducing"]
end
