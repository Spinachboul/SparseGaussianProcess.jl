using Test
using SparseGaussianProcess  # Import the module
using Random

sgp = SGP()# Create an instance of the SGP class
@testset "SparseGaussianProcess.jl" begin

   # Test set_inducing_inputs! function
   @testset "set_inducing_inputs!" begin
      # Define some test data
      X_train = rand(10, 2)
      Y_train = rand(10, 1)
      Z = rand(5, 2)

     
      # Test setting inducing inputs without normalization
      set_inducing_inputs!(sgp, Z)
      @test sgp.nz == size(Z, 1)
      @test !sgp.normalize

      # Test setting inducing inputs with normalization
      set_inducing_inputs!(sgp, Z, true)
      @test sgp.nz == size(Z, 1)
      @test sgp.normalize
   end
end
