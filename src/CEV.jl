using Distributions

function simulation(params, info, N, steps=252, option="call")
    σ = params[1]
    γ = params[2]

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
        for k = 1:steps
            s += s * (r - q) * dt + σ * s^γ * sqrt(dt) * rand(x, 1)[1]
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
parameters = [0.14, 1]
information = [100, 90, 1/12, 0.1, 0]
@time println( simulation(parameters, information, 10000) )
