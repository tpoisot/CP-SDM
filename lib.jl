function _softmax(p)
    w = [exp(p), exp(1-p)]
    return w ./ sum(w)
end

function _no_softmax(p)
    w = [p, 1-p]
    return w
end

function _estimate_q(model, St, Sv; α=0.1, kwargs...)
    train!(model; training=St, kwargs...)
    f̂ = predict(model; threshold=false)[Sv]
    𝐶 = zeros(length(f̂))
    s = _no_softmax.(f̂)
    # Conformal score from the true class label
    for i in eachindex(𝐶)
        𝐶[i] = 1 - (labels(model)[Sv[i]] ? s[i][1] : s[i][2])
    end
    # Get the quantile
    n = length(Sv)
    qᵢ = ceil((n+1)*(1-α))/n
    q̂ = quantile(𝐶, qᵢ)
    return q̂
end

function credibleclasses(prediction, q)
    presence = zeros(prediction, Bool)
    absence = zeros(prediction, Bool)
    for k in keys(prediction)
        s₊, s₋ = _no_softmax(prediction[k])
        if s₊ >= (1-q)
            presence[k] = true
        end
        if s₋ >= (1-q)
            absence[k] = true
        end
    end
    return (presence, absence)
end

# Yah

function cellsize(layer::T; R = 6371.0) where {T <: SDMLayer}
    lonstride, latstride = 2.0 .* stride(layer)
    cells_per_ribbon = 360.0 / lonstride
    latitudes_ranges = LinRange(layer.y[1], layer.y[2], size(layer, 1)+1)
    # We need to express the latitudes in gradients for the top and bottom of each row of
    # cell
    ϕ1 = deg2rad.(latitudes_ranges[1:(end - 1)])
    ϕ2 = deg2rad.(latitudes_ranges[2:end])
    # The difference between the sin of each is what we want to get the area
    Δ = abs.(sin.(ϕ1) .- sin.(ϕ2))
    A = 2π * (R^2.0) .* Δ
    cell_surface = A ./ cells_per_ribbon
    # We then reshape everything to a grid
    surface_grid = reshape(repeat(cell_surface, size(layer, 2)), size(layer))
    # And we return based on the actual type of the input
    S = similar(layer, eltype(surface_grid))
    S.grid .= surface_grid
    return S
end