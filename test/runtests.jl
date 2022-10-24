using MaternKernel
using Test

# Elementary operations 

@testset "Addition" begin
    x = KernelHerdingIterate([0.1, 0.2, 0.7], [0.3, 0.32, 0.35])
    y = x + x
    @test y.weights == [0.2, 0.4, 1.4]
    @test y.vertices == [0.3, 0.32, 0.35]

    w = KernelHerdingIterate([1.0], [0.2])
    z = w + x
    @test z.weights == [1.0, 0.1, 0.2, 0.7]
end

@testset "Scalar multiplication" begin
    x = KernelHerdingIterate([0.1, 0.2, 0.7], [0.3, 0.32, 0.35])
    y = x * 0.5
    @test y.weights == [0.05, 0.1, 0.35]
end

@testset "Subtraction" begin
    x = KernelHerdingIterate([0.1, 0.9], [0.3, 0.35])
    y = KernelHerdingIterate([0.2, 0.8], [0.3, 0.0])
    z = x - y
    @test z.weights == [-0.1, 0.9, -0.8]
    @test z.vertices == [0.3, 0.35, 0.0]
end


# Basic linear algebra

@testset "dot with itself" begin
    x = KernelHerdingIterate([0.5, 0.5], [0., 0.5])
    y = dot(x, x)
    @test y ≈ 0.9548979947844751

    x = KernelHerdingIterate([1.0], [0.])
    y = dot(x, x)
    @assert y ≈ 1.0

    x = KernelHerdingIterate([0.2, 0.3, 0.5], [0.1, 0, 0.7])
    y = dot(x, x)
    @assert y ≈ 0.928316767664454

    x = KernelHerdingIterate([0.5, 0.0, 0.5], [0.1, 0, 0.7])
    y = dot(x, x)
    @assert y ≈ 0.9390493088752212

    x = KernelHerdingIterate([0.5, 0.0, 0.5], [0.1, 0, 0.7])
    w = KernelHerdingIterate([0.2, 0.8], [0.5, 0.0])
    @assert dot(w, x) ≈ dot(x, w) 
    @assert dot(w, x) ≈ 0.9278989673283281
end


@testset "Merging" begin
    x = KernelHerdingIterate([0.1, 0.2, 0.7], [0.3, 0.32, 0.35])
    w = KernelHerdingIterate([1.0], [0.5])
    scalar = 0.5
    MaternKernel.merge_kernel_herding_iterates(x, w, scalar)
    @test x.weights == [0.05, 0.1, 0.35, 0.5]

    x = KernelHerdingIterate([0.1, 0.2, 0.7], [0.3, 0.32, 0.35])
    w = KernelHerdingIterate([1.0], [0.3])
    scalar = 0.5
    MaternKernel.merge_kernel_herding_iterates(x, w, scalar)
    @test x.weights == [0.55, 0.1, 0.35]
end
