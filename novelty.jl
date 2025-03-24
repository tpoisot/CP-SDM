function novelty(historical, projected, var)
    μ = mean.(historical[var])
    σ = std.(historical[var])
    cr_historical = (historical[var] .- μ) ./ σ
    cr_projected = (projected[var] .- μ) ./ σ

    Δclim = similar(cr_historical[1])
    vals = values.(cr_projected)
    
    for position in keys(cr_historical[1])
        dtemp = (cr_historical[1][position] .- vals[1]) .^ 2.0
        dprec = (cr_historical[2][position] .- vals[2]) .^ 2.0
        Δclim[position] = minimum(sqrt.(dtemp .+ dprec))
    end

    return Δclim
end