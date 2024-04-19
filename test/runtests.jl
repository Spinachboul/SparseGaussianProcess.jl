using SparseGaussianProcess
using Test



@testset "SparseGaussianProcess.jl" begin
   # Running tests for inducing points
   @test square_exp([1.0, 2.0], [1.0, 2.0], 1.0) == 1.0

end
