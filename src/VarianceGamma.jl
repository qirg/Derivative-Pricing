include("FFT_European.jl")
using QuadGK
using FastGaussQuadrature
using StatsFuns

function ϕ(u, params, info) # variance gamma characteristic function
    σ = params[1]
    ν = params[2]
    θ = params[3]

    Sₒ = info[1]
    K = info[2]
    t = info[3]
    r = info[4]
    q = info[5]

    Ω = (1 / ν) * log(1 - θ * ν - 0.5 * σ^2 * ν)
    term = exp.(i .* (log(Sₒ) + (r - q + Ω) * t) .* u)
    term .* (1 .- i.*u.*θ.*ν .+ 0.5.*σ^2.*u.^2.*ν).^(-t / ν)
end

parameters = [0.12, 0.2, -0.14]
information = [100, 90, 1/12, 0.1, 0]
@time println( get_value(14, 250, parameters, information, ϕ, 1, "call") )

function integrate(f, a, b)
    nodes, weights = gausslegendre(300)
    x = 0
    for j = 1:length(nodes)
        x += 0.5 * w[j] * f(a + ((nodes[j] + 1)/2) / (1 - (nodes[j] + 1)/2)) * 1/(1 - (nodes[j] + 1)/2)^2
    end
    x
end

function integral(params, info, option="call") # use gauss nodes
    σ = params[1]
    ν = params[2]
    θ = params[3]

    Sₒ = info[1]
    K = info[2]
    t = info[3]
    r = info[4]
    q = info[5]

    function integrand(g)
        d1 = ( log(Sₒ / K) + (r - q + σ^2/2) * g ) / (σ * sqrt(g))
        d2 = d1 - σ * sqrt(g)

        option = lowercase(option)
        if option == "call"
            bs = Sₒ * exp(-q * g) * normcdf(0, 1, d1) - K * exp(-r * g) * normcdf(0, 1, d2)
        elseif option == "put"
            bs = K * exp(-r * g) * normcdf(0, 1, -d2) - Sₒ * exp(-q * g) * normcdf(0, 1, -d1)
        else
            return
        end

        γ = (g^(t/ν - 1) * exp(-g / ν)) / (ν^(t/ν) * gamma(t / ν))

        bs * γ
    end

    function integrate(f, a)
        nodes, w = gausslegendre(10000) # not needed
        x = 0
        for j = 1:length(nodes)
            x += 0.5 * w[j] * f(a + ((nodes[j] + 1)/2) / (1 - (nodes[j] + 1)/2)) * 1/(1 - (nodes[j] + 1)/2)^2
        end
        x
    end
    @time println("estimate: ", integrate(integrand, 0))

    quadgk(integrand, 0, Inf)
end
parameters = [0.12, 0.2, -0.14]
#parameters = [0.14, 0.001, 0.001]
information = [100, 90, 1/12, 0.1, 0]
@time println( integral(parameters, information) )
