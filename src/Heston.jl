include("FFT_European.jl")
using Distributions

function λ(u, params) # κ, θ, σ, ρ, νₒ
    κ = params[1]
    θ = params[2]
    σ = params[3]
    ρ = params[4]
    νₒ = params[5]
    sqrt.( σ^2 .* (u.^2 .+ i.*u) .+ (κ .- i.*ρ.*σ.*u).^2 )
end

function ω(u, params, info)
    κ = params[1]
    θ = params[2]
    σ = params[3]
    ρ = params[4]
    νₒ = params[5]

    Sₒ = info[1]
    K = info[2]
    t = info[3]
    r = info[4]
    q = info[5]

    numerator = exp.( i.*u.*log(Sₒ) .+ i.*u.*(r - q).*t .+ (κ.*θ.*t.*(κ .- i.*ρ.*σ.*u)) ./ σ^2 )
    denominator = (cosh.(λ(u, params).*t./2) .+ (κ .- i.*ρ.*σ.*u) ./ λ(u, params) .* sinh.(λ(u, params).*t./2)).^(2.*κ.*θ ./ σ^2)
    final = numerator ./ denominator
    check = isnan.(final)
    for l in 1:length(final)
        if check[l]
            final[l] = 0 + 0*i
        end
    end
    final
end

function ϕ(u, params, info) # heston model characteristic function
    κ = params[1]
    θ = params[2]
    σ = params[3]
    ρ = params[4]
    νₒ = params[5]

    Sₒ = info[1]
    K = info[2]
    t = info[3]
    r = info[4]
    q = info[5]

    numerator = -(u.^2 .+ i.*u).*νₒ
    denominator = λ(u, params).*coth.(λ(u, params).*t./2) .+ κ .- i.*ρ.*σ.*u
    ω(u, params, info) .* exp.( numerator ./ denominator )
end
parameters = [2, 0.04, 0.5, -0.7, 0.04]
information = [100, 90, 1/2, 0.03, 0]
@time println(get_value(14, 250, parameters, information, ϕ))




function simulation(params, info, N, steps=252, option="call")
    κ = params[1]
    θ = params[2]
    σ = params[3]
    ρ = params[4]
    νₒ = params[5]

    Sₒ = info[1]
    K = info[2]
    t = info[3]
    r = info[4]
    q = info[5]

    dt = t / steps

    payoff = 0
    option = lowercase(option)
    x = Normal() # standard normal

    for j = 1:N
        s = Sₒ
        ν = νₒ
        dw₁ = rand(x, steps)
        dw₂ = ρ .* dw₁ .+ √(1 - ρ^2) .* rand(x, steps)
        for k = 1:steps
            x₁ = dw₁[k]
            x₂ = ρ * x₁ + √(1 - ρ^2) * dw₂[k]
            if ν < 0
                ν = 0
            end
            s += s * (r - q) * dt + √(ν) * s * sqrt(dt) * x₁
            ν += κ * (θ - ν) * dt + σ * √(ν) * sqrt(dt) * x₂
        end
        if option == "call"
            payoff += max(s - K, 0)
        elseif option == "put"
            payoff += max(K - s, 0)
        else
            return
        end
    end
    exp(-r * t) * payoff / N
end
parameters = [2, 0.04, 0.5, -0.7, 0.04]
information = [100, 90, 1/2, 0.03, 0]
@time println( simulation(parameters, information, 10000) )
