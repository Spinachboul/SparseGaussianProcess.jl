using SparseGaussianProcess
using Test



@testset "SparseGaussianProcess.jl" begin
   @test my_f(2,1) == 7
   @test my_f(3,2) == 12
end
