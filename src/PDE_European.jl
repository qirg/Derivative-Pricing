# SDE / PDE / PIDE discretization

# Finite differences
function implicit(Sₒ, K, t, σ, r, q, Δs, Δt, option)
    smax = 2 * Sₒ

    M = convert(Int64, round(smax / Δs))
    ds = smax / M
    N = convert(Int64, round(t / Δt))
    dt = t / N

    price = zeros(M + 1, N + 1)
    value = collect(0:ds:smax)

    index = collect(0:M)

    option = lowercase(option)
    if option == "call"
        price[:, N + 1] = max.(value .- K, 0)
        price[M + 1, :] = (value[end] - K) .* exp.(-r .* dt .* (N .- collect(0:N)))
        price[1, :] = zeros(N + 1)
    elseif option == "put"
        price[:, N + 1] = max.(K .- value, 0)
        price[1, :] = (K - value[1]) .* exp.(-r .* dt .* (N .- collect(0:N)))
        price[M + 1, :] = zeros(N + 1)
    else
        return
    end

    a = 0.5 .* (r .* dt .* index .- σ^2 .* dt .* index.^2) # l
    b = 1 .+ σ^2 .* dt .* index.^2 .+ r .* dt # a
    c = -0.5 .* (r .* dt .* index .+ σ^2 .* dt .* index.^2) # u

    # tridiagonal matrix
    A = diagm(a[3:M], -1) + diagm(b[2:M]) + diagm(c[2:(M-1)], 1)
    @assert norm(inv(A), Inf) <= 1

    L, U, p = lu(A)

    aux = zeros(M - 1)

    for j = N:-1:1
        aux[1] = a[2] * price[1, j + 1]
        aux[M - 1] = c[M + 1] * price[M + 1, j + 1]
        price[2:M, j] = inv(U) * (inv(L) * (price[2:M, j + 1] .- aux))
    end
    price[:, 1][div((M + 2), 2)]
end
@time println( implicit(50., 50., 5/12, 0.4, 0.1, 0., 0.5, 5/2400, "put") )
@time println( implicit(50., 50., 5/12, 0.4, 0.1, 0., 0.5, 5/2400, "call") )

function explicit(Sₒ, K, t, σ, r, q, Δs, Δt, option)
    smax = 2 * Sₒ

    M = convert(Int64, round(smax / Δs))
    ds = smax / M
    N = convert(Int64, round(t / Δt))
    dt = t / N

    price = zeros(M + 1, N + 1)
    value = collect(0:ds:smax)

    index = collect(0:M)

    option = lowercase(option)
    if option == "call"
        price[:, N + 1] = max.(value .- K, 0)
        price[M + 1, :] = (value[end] - K) .* exp.(-r .* dt .* (N .- collect(0:N)))
        price[1, :] = zeros(N + 1)
    elseif option == "put"
        price[:, N + 1] = max.(K .- value, 0)
        price[1, :] = (K - value[1]) .* exp.(-r .* dt .* (N .- collect(0:N)))
        price[M + 1, :] = zeros(N + 1)
    else
        return
    end

    a = 0.5 .* dt .* (σ^2 .* index.^2 .- r .* index) # l
    b = 1 .- dt .* (σ^2 .* index.^2 .+ r) # a
    c = 0.5 .* dt .* (σ^2 .* index.^2 .+ r .* index) # u

    # tridiagonal matrix
    A = diagm(a[3:M], -1) + diagm(b[2:M]) + diagm(c[2:(M-1)], 1)
    #@assert maximum(eigvals(A)) < 1
    @assert norm(A, Inf) < 1

    aux = zeros(M - 1)

    for j = (N+1):-1:2
        aux[1] = a[2] * price[1, j]
        aux[M - 1] = c[M + 1] * price[M + 1, j]
        price[2:M, j - 1] = A * price[2:M, j] .+ aux
    end
    price[:, 1][div((M + 2), 2)]
end
@time println( explicit(50., 50., 5/12, 0.4, 0.1, 0., 2, 5/2400, "put") )
@time println( explicit(50., 50., 5/12, 0.4, 0.1, 0., 2, 5/2400, "call") )

function crank_nicolson(Sₒ, K, t, σ, r, q, Δs, Δt, option)
    smax = 2 * Sₒ

    M = convert(Int64, round(smax / Δs))
    ds = smax / M
    N = convert(Int64, round(t / Δt))
    dt = t / N

    price = zeros(M + 1, N + 1)
    value = collect(0:ds:smax)

    index = collect(0:M)

    option = lowercase(option)
    if option == "call"
        price[:, N + 1] = max.(value .- K, 0)
        price[M + 1, :] = (value[end] - K) .* exp.(-r .* dt .* (N .- collect(0:N)))
        price[1, :] = zeros(N + 1)
    elseif option == "put"
        price[:, N + 1] = max.(K .- value, 0)
        price[1, :] = (K - value[1]) .* exp.(-r .* dt .* (N .- collect(0:N)))
        price[M + 1, :] = zeros(N + 1)
    else
        return
    end

    a = 0.25 .* dt .* (σ^2 .* index.^2 .- r .* index)
    b = 1 .- 0.5 .* dt .* (σ^2 .* index.^2 .+ r)
    c = 0.25 .* dt .* (σ^2 .* index.^2 .+ r .* index)

    A1 = diagm(-a[3:M], -1) + diagm(1 .- b[2:M] .+ 1) + diagm(-c[2:(M-1)], 1)
    A2 = diagm(a[3:M], -1) + diagm(1 .+ b[2:M] .- 1) + diagm(c[2:(M-1)], 1)
    @assert norm(inv(A1) * A2, Inf) < 1

    L, U, p = lu(A1)

    aux = zeros(M - 1)

    for j = N:-1:1
        aux[1] = a[2] * (price[1, j] + price[1, j + 1])
        aux[M - 1] = c[M + 1] * (price[M + 1, j] + price[M + 1, j + 1])
        price[2:M, j] = inv(U) * (inv(L) * (A2 * price[2:M, j + 1] .+ aux))
    end
    price[:, 1][div((M + 2), 2)]
end
@time println( crank_nicolson(50., 50., 5/12, 0.4, 0.1, 0., 2, 5/2400, "put") )
@time println( crank_nicolson(50., 50., 5/12, 0.4, 0.1, 0., 2, 5/2400, "call") )
