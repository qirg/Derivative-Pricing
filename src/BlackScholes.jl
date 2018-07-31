include("FFT_European.jl")
using StatsFuns

function ϕ(u, params, info) # black-scholes characteristic function
    σ = params[1]

    Sₒ = info[1]
    K = info[2]
    t = info[3]
    r = info[4]
    q = info[5]

    exp.(i * (log(Sₒ) + (r - q - 0.5 * σ^2) * t) .* u .- (0.5 .* σ^2 .* u.^2 .* t))
end

parameters = [0.14]
information = [100, 90, 1/12, 0.1, 0]
@time println( get_value(14, 250, parameters, information, ϕ) )
@time println( get_value(14, 250, parameters, information, ϕ, -5, "put") )


function closed_form(params, info, option="call")
    σ = params[1]

    Sₒ = info[1]
    K = info[2]
    t = info[3]
    r = info[4]
    q = info[5]

    d1 = ( log(Sₒ / K) + (r - q + σ^2/2)*t ) / (σ * sqrt(t))
    d2 = d1 - σ * sqrt(t)

    option = lowercase(option)
    if option == "call"
        Sₒ * exp(-q * t) * normcdf(0, 1, d1) - K * exp(-r * t) * normcdf(0, 1, d2)
    elseif option == "put"
        K * exp(-r * t) * normcdf(0, 1, -d2) - Sₒ * exp(-q * t) * normcdf(0, 1, -d1)
    else
        return
    end
end
parameters = [0.14]
information = [100, 90, 1/12, 0.1, 0]
@time println( closed_form(parameters, information, "call") )

#=
@everywhere function simulation(params, info, N, steps=252, option="call")
    σ = params[1]

    Sₒ = info[1]
    K = info[2]
    t = info[3]
    r = info[4]
    q = info[5]

    dt = t / steps

    payoff = 0
    option = lowercase(option)
    rng = MersenneTwister(3579)

    for j = 1:N
        s = Sₒ
        dw₁ = randn!(rng, zeros(steps))
        for k = 1:steps
            s += s * ( (r - q) * dt + σ * sqrt(dt) * dw₁[k] )
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
parameters = [0.14]
information = [100, 90, 1/12, 0.1, 0]
@time println( simulation(parameters, information, 10000) )

addprocs(2)
#rmprocs()
function parallel_simulation(params, info, N, steps=252, option="call", ncores=2)
    value = @parallel (+) for i=1:ncores
        simulation(params, info, ceil(Int, N / ncores), steps, option)
    end
    value / ncores
end
parameters = [0.14]
information = [100, 90, 1/12, 0.1, 0]
@time println( parallel_simulation(parameters, information, 10000) )
parameters = [0.29]
information = [164, 165, 0.0959, 0.0521, 0]
@time println( parallel_simulation(parameters, information, 10000) )
=#

function binomial_tree(params, info, n, option="call")
    σ = params[1]

    Sₒ = info[1]
    K = info[2]
    t = info[3]
    r = info[4]
    δ = info[5]

    if n == 0
        return 0
    else
        Δ = t / n
    end
    discount = exp(-r * Δ)
    u = exp(σ * √(Δ))
    d = 1 / u
    p = ( exp((r - δ) * Δ) - d ) / ( u - d )

    move = d .^ (n:-1:0) .* u .^ (0:n)

    option = lowercase(option)
    if option == "call"
        leaf = max.(Sₒ .* move .- K, 0)
    elseif option == "put"
        leaf = max.(K .- Sₒ .* move, 0)
    else
        return
    end

    for i in (n:-1:1)
        move = (d .^ ((i-1):-1:0) .* u .^ (0:(i-1)))
        if option == "call" # only for American options
            parent = max.(Sₒ .* move .- K, 0)
        elseif option == "put"
            parent = max.(K .- Sₒ .* move, 0)
        else
            return
        end
        leaf = discount .* (p .* leaf[2:end] .+ (1 - p) .* leaf[1:(end-1)])
        leaf = max.(leaf, parent) # only for American options
    end
    leaf[1]
end
parameters = [0.14]
information = [100, 90, 1/12, 0.1, 0]
@time println(binomial_tree(parameters, information, 1000))

parameters = [0.3]
information = [5, 10, 1, 0.06, 0]
@time println(binomial_tree(parameters, information, 256, "put")) # compare with matlab speed paper
