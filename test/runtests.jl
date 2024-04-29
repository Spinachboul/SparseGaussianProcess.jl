using Test
using SparseGaussianProcess  # Import your package module

# Test set_inducing_inputs!
@testset "set_inducing_inputs!" begin
    sgp = SparseGaussianProcess.SGP()
    X = rand(10, 3)  # Example training data
    SparseGaussianProcess.set_inducing_inputs!(sgp, X)

    @test !isnothing(sgp.Z)
    @test size(sgp.Z, 2) == size(X, 2)
    @test sgp.nz == sgp.options["n_inducing"]


@testset "SparseGaussianProcess.jl" begin
   @test my_f(2,1) == 7
   @test my_f(3,2) == 12
end