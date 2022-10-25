using FrankWolfe
using LinearAlgebra
"""
Kernel herding iterate.
"""
struct KernelHerdingIterate{T}
    weights::Vector{T}
    vertices::Vector{T}
end


LinearAlgebra.dot(y, x::KernelHerdingIterate) = LinearAlgebra.dot(x, y)

"""
Addition for KernelHerdingIterate.
"""
function Base.:+(x1::KernelHerdingIterate, x2::KernelHerdingIterate)
    x = copy(x1)
    for idx2 in eachindex(x2.vertices)
        p2 = x2.vertices[idx2]
        found = false
        for idx in eachindex(x.vertices)
            p = x.vertices[idx]
            if p≈p2
                x.weights[idx] += x2.weights[idx2]
                found = true
            end
        end
        if !found
            push!(x.weights, x2.weights[idx2])
            push!(x.vertices, x2.vertices[idx2])
        end
    end
    return x
end

"""
Multiplication for KernelHerdingIterate.
"""
function Base.:*(x::KernelHerdingIterate, scalar::Real)
    w = copy(x)
    w.weights .*= scalar
    return w
end

"""
Multiplication for KernelHerdingIterate, different order.
"""
function Base.:*(scalar::Real, x::KernelHerdingIterate)
    w = copy(x)
    w.weights .*= scalar
    return w
end

"""
Subtraction for KernelHerdingIterate.
"""
function Base.:-(x1::KernelHerdingIterate, x2::KernelHerdingIterate)
    return x1 + (-1)*x2
end

# # Different types of MeanElements

"""
MeanElement μ must implement dot with a functional.
"""
abstract type MeanElement
end

"""
μ for the uniform distribution.
"""
struct UniformMeanElement <: MeanElement
end

"""
μ for the nonuniform distribution.
"""
struct NonUniformMeanElement{T} <: MeanElement
    distribution::Vector{T}
end

# # Scalar products, norms, and auxiliary functions

"""
Scalar product for two KernelHerdingIterates.
"""
function LinearAlgebra.dot(x1::KernelHerdingIterate, x2::KernelHerdingIterate)
    w_1 = x1.weights
    w_2 = x2.weights
    v_1 = x1.vertices
    v_2 = x2.vertices
    w_matrix = w_1 * w_2'
    p_matrix = [kernel_evaluation.(v_1[i], v_2[j]) for i in eachindex(v_1), j in eachindex(v_2)]
    scalar_product_matrices = dot(w_matrix, p_matrix)
    return scalar_product_matrices
end

"""
Scalar product for KernelHerdingIterate with UniformMeanElement.
"""
function LinearAlgebra.dot(x::KernelHerdingIterate{T}, mu::UniformMeanElement) where T
    weights = x.weights
    vertices = x.vertices
    value = 0
    for j in eachindex(weights)
        value += weights[j] * compute_integral(vertices[j])
    end
    return value
end

function compute_integral(y)
    return .5 *(exp(-y - 1) * (exp(2 * y) * (y - 3) + 4 * exp(y + 1) - y - 3) )
end


"""
Norm of UniformMeanElement, corresponds to ||µ||.
"""
function LinearAlgebra.norm(mu::UniformMeanElement)
    return sqrt(1/4 * (2 + 10/ ℯ^2))
end

"""
Scalar product for KernelHerdingIterate with NonUniformMeanElement.
"""
# function LinearAlgebra.dot(x::KernelHerdingIterate{T}, mu::NonUniformMeanElement) where T
#     weights = x.weights
#     vertices = x.vertices
#     p = mu.distribution
#     value = 0
#     for i in eachindex(p)
#         for j in eachindex(weights)
#             value += weights[j] * compute_integral(vertices[j]) 
#         end
#         value *= p[i]
#     end
#     return value
# end

"""
Norm of NonUniformMeanElement, corresponds to ||µ||.
"""
# function LinearAlgebra.norm(mu::NonUniformMeanElement)
    # p = mu.distribution
    # value = 0
    # for i in eachindex(p)
    #     for j in eachindex(p)
    #         value += p[j]
    #     end
    #     value += p[i]
    # end
    # return sqrt(value * (2 + 10/ ℯ^2))

# end



# # Technical replacements for the infinite-dimensional kernel herding setting

function Base.similar(x::KernelHerdingIterate, ::Type{T}) where T
    return KernelHerdingIterate(similar(x.weights, T), similar(x.vertices, T))
end

function Base.similar(x::KernelHerdingIterate{T}) where T
    return Base.similar(x, T)
end

function Base.eltype(x::KernelHerdingIterate{T}) where T
    return T
end

function FrankWolfe.compute_active_set_iterate!(active_set::FrankWolfe.ActiveSet{AT,R,IT}) where {AT, R, IT <: KernelHerdingIterate}

    empty!(active_set.x.weights)
    empty!(active_set.x.vertices)

    for idx in eachindex(active_set)
        push!(active_set.x.weights, active_set.weights[idx])
        push!(active_set.x.vertices, only(active_set.atoms[idx].vertices))
    end
end

function Base.empty!(active_set::FrankWolfe.ActiveSet{AT,R,IT}) where {AT, R, IT <: KernelHerdingIterate}
    empty!(active_set.atoms)
    empty!(active_set.weights)
    empty!(active_set.x.weights)
    empty!(active_set.x.vertices)
    push!(active_set.x.weights, 1)
    push!(active_set.x.vertices, 0)
    return active_set
end

function FrankWolfe.active_set_initialize!(active_set::FrankWolfe.ActiveSet{AT,R,IT}, v::KernelHerdingIterate) where {AT, R, IT <: KernelHerdingIterate}
    empty!(active_set)
    push!(active_set, (one(R), v))
    FrankWolfe.compute_active_set_iterate!(active_set)
    return active_set
end

function FrankWolfe.active_set_update!(active_set::FrankWolfe.ActiveSet{AT, R, IT}, lambda, atom, renorm=true, idx=nothing) where {AT, R, IT <: KernelHerdingIterate}
    
    # rescale active set
    active_set.weights .*= (1 - lambda)
    # add value for new atom
    if idx === nothing
        idx = FrankWolfe.find_atom(active_set, atom)
    end
    updating = false
    if idx > 0
        @inbounds active_set.weights[idx] = active_set.weights[idx] + lambda
        updating = true
    else
        push!(active_set, (lambda, atom))
    end
    if renorm
        FrankWolfe.active_set_cleanup!(active_set, update=false)
        FrankWolfe.active_set_renormalize!(active_set)
    end
    merge_kernel_herding_iterates(active_set.x, atom, lambda)
    return active_set
end

function FrankWolfe.active_set_update_iterate_pairwise!(xold::KernelHerdingIterate, lambda, fw_atom, away_atom)
    x = xold + lambda * fw_atom - lambda * away_atom
    copy!(xold, x)
    return xold
end

function merge_kernel_herding_iterates(x1::KernelHerdingIterate, x2::KernelHerdingIterate, scalar)
    @assert(0 <= scalar <= 1, "Scalar should be in [0, 1].")
    if scalar == 0
        return x1
    elseif scalar == 1
        return x2
    else
        x1.weights .*= (1 - scalar)
        for idx in eachindex(x2.weights)
            p_2 = x2.vertices[idx]
            found = false
            for idx2 in eachindex(x1.weights)
                p_1 = x1.vertices[idx2]
                if p_2 ≈ p_1
                    x1.weights[idx2] += x2.weights[idx] * scalar
                    found = true
                end
            end
            if !found
                push!(x1.weights, x2.weights[idx]*scalar)
                push!(x1.vertices, x2.vertices[idx])
            end
        end
    end
@assert sum(x1.weights) ≈ 1
@assert all(-1 .<= x1.vertices .<=1)
end


function Base.copy(x::KernelHerdingIterate{T}) where T
    return KernelHerdingIterate(copy(x.weights), copy(x.vertices))
end

function Base.copy!(x::KernelHerdingIterate{T}, y::KernelHerdingIterate{T}) where T
    copy!(x.weights, y.weights)
    copy!(x.vertices, y.vertices)
end

# Gradient, loss, and extreme point computations

"""
The gradient < x - μ, . > is represented by x and μ.
"""
mutable struct KernelHerdingGradient{T, D <: MeanElement}
    x:: KernelHerdingIterate{T}
    mu:: D
end

"""
Scalar product for KernelHerdingIterate with KernelHerdingGradient.
"""
function LinearAlgebra.dot(x::KernelHerdingIterate, g::KernelHerdingGradient)
    scalar_product = dot(g.x, x)
    scalar_product -= dot(g.mu, x)
    return scalar_product
end

"""
Creates the loss function and the gradient function, given a MeanElement μ, that is,
    1/2 || x - μ ||_H²      and     < x - μ, . >,
respectively.
"""
function create_loss_function_gradient(mu::MeanElement)
    
    mu_squared = norm(mu)^2
    function evaluate_loss(x::KernelHerdingIterate)
        l = dot(x, x)
        l += mu_squared
        l -= 2 * dot(x, mu)
        l /= 2
        return l
    end
    function evaluate_gradient(g::KernelHerdingGradient, x::KernelHerdingIterate)
        g.x = x
        g.mu = mu
    return g
    end
    return evaluate_loss, evaluate_gradient
end

"""
The marginal polytope of the Wahba kernel.
"""
struct MarginalPolytope <: FrankWolfe.LinearMinimizationOracle
    number_iterations:: Int
end

"""
Computes the extreme point in the Frank-Wolfe algorithm for kernel herding in Wahba's kernel.
"""
function FrankWolfe.compute_extreme_point(lmo::MarginalPolytope, direction::KernelHerdingGradient; kw...)
    optimal_value = Inf
    optimal_vertex = nothing
    current_vertex = nothing
    for iteration in 1:lmo.number_iterations 
        current_vertex = KernelHerdingIterate([1.0], [-1 + (2 * iteration -1) / (lmo.number_iterations)])
        current_value = dot(direction, current_vertex)
        if current_value < optimal_value
            optimal_vertex = current_vertex
            optimal_value = current_value
        end
    end
    @assert(optimal_vertex !== nothing, "This should never happen.")
    @assert(-1 <= only(optimal_vertex.vertices) <= 1, "Vertices have to correspond to real numbers in [0, 1].")
    return optimal_vertex
end


"""
Evaluates the Matern kernel over two real numbers.
"""

function kernel_evaluation(y1, y2)
    return (1+norm(y1-y2)) * exp(-norm(y1-y2))
end


function get_distribution(k)
    p = rand(Float64, k)
    distribution = p / (2 * sum(p))
    @assert(sum(distribution) ≈ 0.5, "The vector p still does not correspond to a distribution.")
    return distribution
end


function get_Riemann_sum(t)
    val = 0
    for i in 1:t
        for j in 0:t-1
            val += kernel_evaluation(-1+(2*i+1)/t, -1+(2*j+1)/t)
        end
    end
    return 1/t^2 * val
end

