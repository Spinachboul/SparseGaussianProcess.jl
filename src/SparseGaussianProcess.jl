module SparseGaussianProcess
using LinearAlgebra
using Random

abstract type AbstractSurrogateModel end

mutable struct SGP <: AbstractSurrogateModel
    name::String
    Z::Union{Nothing, Matrix{Float64}}
    woodbury_data::Dict{String, Union{Nothing, Vector{Float64}, Matrix{Float64}}}
    optimal_par::Dict{String, Union{Nothing, Vector{Float64}}}
    optimal_noise::Union{Nothing, Float64}
    options::Dict{String, Any}
    supports::Dict{String, Bool}
    training_points::Dict{String, Any}
    nt::Int
    nx::Int
    nz::Int
    normalize::Bool
    is_continuous::Bool
    _correlation_types::Dict{String, Function}

    function SGP()
        name = "SparseGaussianProcess"
        Z = nothing
        woodbury_data = Dict("vec" => nothing, "inv" => nothing)
        optimal_par = Dict()
        optimal_noise = nothing
        options = Dict(
            "corr" => "square_exp",
            "poly" => "constant",
            "theta_bounds" => [1e-6, 1e2],
            "noise0" => [1e-2],
            "hyper_opt" => "Cobyla",
            "eval_noise" => true,
            "nugget" => 1000.0 * eps(Float64),
            "method" => "FITC",
            "n_inducing" => 10,
            "use_hetero_noise" => false  # Fixed the issue by removing spaces
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
        _correlation_types = Dict{String, Function}()
        new(name, Z, woodbury_data, optimal_par, optimal_noise, options, supports, training_points, nt, nx, nz, normalize, is_continuous, _correlation_types)
    end
end

function set_inducing_inputs!(sgp::SGP, Z::Union{Nothing, Matrix{Float64}}, normalize::Bool=false)
    if isnothing(Z)
        # Initialize sgp.Z with random training data
        X = sgp.training_points["X"]
        random_idx = randperm(sgp.nt)[1:sgp.options["n_inducing"]]
        sgp.Z = X[random_idx, :]
    else
        if size(Z, 2) != sgp.nx
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
        sgp.nz = size(sgp.Z, 1)
    end
end

function _new_train!(sgp::SGP)
    if isnothing(sgp.Z)
        set_inducing_inputs!(sgp, nothing)
    end

     # Make sure the latent function is scalars
     Y = sgp.training_points["Y"]
     _, output_dim = size(Y)
    if output_dim != 1
        throw(DimensionMismatch("DimensionError: Y.shape[2] != 1"))
    end

    # Make sure the noise is not hetero
    if sgp.options["use hetero noise"]
        throw(NotImplementedError("ArgumentError: Heteroscedastic noise is not supported"))
    end

    # Make sure we are using continuous vatiables only
    if !sgp.is_continuous
        throw(NotImplementedError("SGP does not support mixed-Integer variables"))
    end

    # Works only with COBYLA because of no likelihood gradients
    if sgp.options["hyper_opt"] != "Cobyla"
        throw(NotImplementedError("SGP works only with COBYLA internal optimizer"))

    end

    return _new_train(sgp)
end


function _reduced_likelihood(sgp::SGP, theta::Vector{Float64}, noise::Float64)
    X = sgp.training_points["X"]
    Y = sgp.training_points["Y"]
    Z = sgp.Z

    if sgp.options["eval_noise"]
        sigma2 = theta[end-1]
        noise = theta[end]
        theta = theta[1:end-2]
    else
        sigma2 = theta[end]
        noise = sgp.options["noise0"]
        theta = theta[1:end - 1]
        
    end

    nugget = sgp.options["nugget"]

    if sgp.options["method"] == "VFE"
        likelihood, w_vec, w_inv = _vfe(sgp, X, Y, Z, theta, sigma2, noise, nugget)
    else
        likelihood, w_vec, w_inv = _fitc(sgp, X, Y, Z, theta, sigma2, noise, nugget)
    end
    
    sgp.woodbury_data["vec"] = w_vec
    sgp.woodbury_data["inv"] = w_inv
    
    params = Dict("theta" => theta, "sigma2" => sigma2)
    
    return likelihood, params
end


function _reduced_likelihood_gradient(sgp::SGP, theta::Vector{Float64})
    throw(NotImplementedError("SGP gradient of reduced likelihood not implemented yet"))
end

function _reduced_likelihood_hessian(sgp::SGP, theta::Vector{Float64})
    throw(NotImplementedError("SGP hessian of reduced likelihood not implemented yet"))
end

function _compute_K(sgp::SGP, A::Matrix{Float64}, B::Matrix{Float64}, theta::Vector{Float64}, sigma2::Float64)
    dx = differences(A, B)
    d = _componentwise_distance(sgp, dx)
    r = sgp._correlation_types[sgp.options["corr"]](theta, d)
    R = reshape(r, size(A, 1), size(B, 1))
    K = sigma2 * R
    return K
end

function _fitc(sgp::SGP, X::Matrix{Float64}, Y::Matrix{Float64}, Z::Matrix{Float64}, theta::Vector{Float64}, sigma2::Float64, noise::Float64, nugget::Float64)
    Knn = fill(sigma2, sgp.nt)
    Kmm = _compute_K(sgp, Z, Z, theta, sigma2)
end

end  # Module

# Export the functions
export SGP, set_inducing_inputs!, _new_train!, _reduced_likelihood