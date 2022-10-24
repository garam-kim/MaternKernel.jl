module MaternKernel
using FrankWolfe
using LinearAlgebra

include("iterates.jl")
export KernelHerdingIterate, KernelHerdingGradient, MarginalPolytope, create_loss_function_gradient, UniformMeanElement, NonUniformMeanElement, get_distribution
end