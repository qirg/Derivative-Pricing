# Arithmetic Asian Options

#Xₒ: Initial Asset Price
#K: Strike Price
#σ: volatility
#T: Time to expiry
#r: risk-free rate
#m: number of steps
#simulations: number of simulations

# Average Price
function asian_call_price(Xₒ, K, σ, T, r, m)
    x̂ = 0.0
    X::Float64 = Xₒ
    Δ = T / m
    for i in 1:m
        X *= exp( (r - σ^2/2) * Δ + σ * √(Δ) * randn() )
        x̂ += X
    end
    return exp(-r * T) * max(x̂/m - K, 0)
end

# Average Price
function asian_put_price(Xₒ, K, σ, T, r, m)
    x̂ = 0.0
    X::Float64 = Xₒ
    Δ = T / m
    for i in 1:m
        X *= exp( (r - σ^2/2) * Δ + σ * √(Δ) * randn() )
        x̂ += X
    end
    return exp(-r * T) * max(K - x̂/m, 0)
end

# Average Strike
function asian_call_strike(Xₒ, S, σ, T, r, m)
    x̂ = 0.0
    X::Float64 = Xₒ
    Δ = T / m
    for i in 1:m
        X *= exp( (r - σ^2/2) * Δ + σ * √(Δ) * randn() )
        x̂ += X
    end
    return exp(-r * T) * max(S - x̂/m, 0)
end

# Average Strike
function asian_put_strike(Xₒ, S, σ, T, r, m)
    x̂ = 0.0
    X::Float64 = Xₒ
    Δ = T / m
    for i in 1:m
        X *= exp( (r - σ^2/2) * Δ + σ * √(Δ) * randn() )
        x̂ += X
    end
    return exp(-r * T) * max(x̂/m - S, 0)
end

# Example
function f_call_price(simulations)
    return mean([asian_call_price(30, 30, 0.025, 1, 0.05, 252) for i in 1:simulations])
end

function f_put_price(simulations)
    return mean([asian_put_price(30, 30, 0.025, 1, 0.05, 252) for i in 1:simulations])
end

function f_call_strike(simulations)
    return mean([asian_call_strike(30, 30, 0.025, 1, 0.05, 252) for i in 1:simulations])
end

function f_put_strike(simulations)
    return mean([asian_put_strike(30, 30, 0.025, 1, 0.05, 252) for i in 1:simulations])
end

println("Example 1")
@time println(f_call_price(10000))
@time println(f_put_price(10000))
@time println(f_call_strike(10000))
@time println(f_put_strike(10000))

#using IJulia
#notebook()
