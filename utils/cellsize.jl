
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