#=
Heston
Variance Gamma
=#
i = 1im

function ψ(u, params, info, ϕ, α=1, option="call")
    option = lowercase(option)
    if option == "call"
        # Must be positive for a call
        @assert α > 0
    elseif option == "put"
        # Must be negative for a put
        @assert α < 0
    end

    Sₒ = info[1]
    K = info[2]
    t = info[3] # need
    r = info[4] # need
    q = info[5]

    exp(-r * t) ./ ((α .+ i.*u) .* (α .+ i.*u .+ 1)) .* ϕ(u .- (α .+ 1).*i, params, info)
end


function get_value(n, B, params, info, ϕ, α=1, option="call")
    s = info[1]
    k = info[2]
    t = info[3]
    r = info[4]
    q = info[5]

    #α = 1
    N = 2^n
    Δk = B / N # η
    Δν = 2 * π / (N * Δk)

    ν = [(j - 1) * Δν for j in 1:N] # u array

    z = zeros(N)
    z[1] = 1
    w = [Δν / 2 * (2 - z[j]) for j in 1:length(ν)] # weight function

    #β = log(s) - Δk * N / 2
    β = log(k) - Δk * N / 2
    m = 1:N
    kₒ = β .+ (m .- 1) .* Δk

    option = lowercase(option)
    if option == "call"
        # Must be positive for a call
        @assert α > 0
    elseif option == "put"
        # Must be negative for a put
        @assert α < 0
    end

    x = exp.(-i .* β .* ν) .* ψ(ν, params, info, ϕ, α, option) .* w
    y = fft(x)

    prices = exp.(-α .* kₒ) ./ π .* real.(y)
    mid = convert(Int64, N/2)
    prices[mid + 1]
end
