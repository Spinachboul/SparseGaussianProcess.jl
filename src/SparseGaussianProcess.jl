module SparseGaussianProcess
using LinearAlgebra
using Random

abstract type AbstractSurrogateModel end

mutable struct SGP <: AbstractSurrogateModel
    name::String
    Z::Union{Nothing, Matrix{Float64}}
    woodbury_data::Dict{String,Union{Nothing,Vector{Float64}, Matrix{Float64}}}
    optimal_par:: Dict{String, Union{Nothing, Vector{Float64}}}
    optimal_noise:: Union{Nothing, Float64}
    options::Dict{String, Any}
    supports::Dict{String, Bool}
    training_points::Dict{String, Any}
    nt::Int
    nx::Int
    nz::Int
    normalize::Bool
    is_continuous::Bool
    _correlation_types::Dict{String, Function}

    function square_exp(x1::Vector{T}, x2::Vector{T}, theta::T) where T
        sqdist = sum((x1 .- x2).^2)
        return exp(-sqdist / (2 * theta^2))
    end

    export square_exp()
    
    function SGP()
        name = "SparseGaussianProcess"
        Z = nothing
        woodbury_data = Dict("vec" => nothing, "inv" => nothing)
        optimal_par = Dict()
        optimal_noise = nothing
        options = Dict(
            "corr" => "square_exp", # guassian kernel only
            "poly" => "constant", # constant mean only
            "theta_bounds" => [1e-6, 1e2], # Upper bound increased as compared to Kriging based one
            "noise0" => [1e-2], # Gaussian noise on observed training data
            "hyper_opt" => "Cobyla", # Optimizer for Hyper parameter optimization
            "eval_noise" => true, # For SGP evaluate noise by default
            "nugget" => 1000.0 * eps(Float64), # Slightly increased as compared to kriging based one
            "method" => "FITC", # Method used by Sparse GP model
            "n_inducing" => 10, # Number of inducing points
        )
        supports = Dict(
            "derivatives" => false,
            "variances" => true,
            "variance_derivatives" => false,
            "x_hierarchy" => false
        )
        training_points = Dict()
        nt = 0
        nx = 0
        nz = 0
        normalize = false
        is_continuous = true
        _correlation_types = Dict(
            "square_exp" => square_exp,
            "matern32" => matern32,
            "matern52" => matern52,
            "cubic" => cubic,
            "linear" => linear,
            "constant" => constant
        )
        new(name, Z, woodbury_data, optimal_par, optimal_noise, options, supports, training_points, nt, nx, nz, normalize, is_continuous, _correlation_types)
    end
end

function set_inducing_inputs!(sgp::SGP, Z::Union{Nothing, Matrix{Float64}}, normalize::Bool=false)
    if isnothing(Z)
        sgp.Z = sgp.options["n_inducing"]
        X = sgp.training_points["X"]
        random_idx = randperm(sgp.nt)[1:sgp.nz]
        sgp,Z = X[random_idx,:]
    else
        if size(Z,2) != sgp.nx
            throw(DimensionMismatch("DimensionError: Z.shape[2] != X.shape[2]"))
        end
        sgp.Z = copy(Z)
        if normalize
            X = sgp.training_points["X"]
            y = sgp.training_points["y"]
            sgp.normalize = true
            _, _, X_offset, _, X_scale, _ = standardization(X, y)
            sgp.Z = (sgp.Z .- X_offset') ./ X_scale'
        else
            sgp.normalize = false
        end
        sgp.nz = size(sgp.Z,1)
    end
end

end # module

# Export the functions
export set_inducing_inputs()
export SGP()            

