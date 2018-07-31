include("VarianceGamma.jl")
using Base.Test

parameters = [0.12, 0.2, -0.14]
information = [100, 90, 1/12, 0.1, 0]
@testset "VG FFT Tests" begin
    @test get_value(14, 250, parameters, information, ϕ, 1, "call") ≈ 10.828714422701717
    @test get_value(14, 250, parameters, information, ϕ, -5, "put") ≈ 0.08190892784047768
end
