using SpeciesDistributionToolkit
using CairoMakie
using Statistics

# Functions

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
    qᵢ = (n+1)*(1-α)/n
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
    latitudes_ranges = (layer.y[1]):latstride:(layer.y[2])
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

# Illustration

CHE = SpeciesDistributionToolkit.openstreetmap("Corse")
bio_vars = collect(1:19)
provider = RasterData(CHELSA2, BioClim)
L = SDMLayer{Float32}[
    SDMLayer(
        provider;
        layer = x,
        SpeciesDistributionToolkit.boundingbox(CHE)...,
    ) for x in bio_vars
];
mask!(L, CHE)
sp = taxon("Sitta whiteheadi")
presences = occurrences(
    sp,
    first(L),
    "occurrenceStatus" => "PRESENT",
    "limit" => 300
)
while length(presences) < min(count(presences), 2_000)
    occurrences!(presences)
end

presencelayer = mask(first(L), presences)
background = pseudoabsencemask(DistanceToEvent, presencelayer)
bgpoints = backgroundpoints(nodata(background, d -> d < 2), 3sum(presencelayer))

# Set up the model
sdm = SDM(ZScore, NaiveBayes, L, presencelayer, bgpoints)
hyperparameters!(classifier(sdm), :η, 1e-3);
hyperparameters!(classifier(sdm), :interactions, :all);
hyperparameters!(classifier(sdm), :epochs, 5_000);

# Train the model with optimal set of variables
variables!(sdm, ForwardSelection; included=[1,12], verbose=true)

ConfusionMatrix(sdm) |> mcc

prd = predict(sdm, L; threshold=false)
heatmap(prd)

cs = cellsize(prd)

cmodel = deepcopy(sdm)
q = _estimate_q(cmodel, holdout(cmodel)...; α=0.05)

rlevels = LinRange(0.01, 0.2, 25)
qs = [_estimate_q(cmodel, holdout(cmodel)...; α=u) for u in rlevels]
scatter(rlevels, qs)

Cp, Ca = credibleclasses(prd, q)
heatmap(Ca .& Cp, colorrange=(0, 1))

poly(CHE, color=:white, strokecolor=:black, strokewidth=1)
heatmap!(nodata(Cp, false), colormap=[:transparent, :grey80])
heatmap!(nodata(Ca .& Cp, false), colormap=[:transparent, :grey20])
scatter!(presencelayer, markersize=3, color=:orange)
#contour!(predict(sdm, L); color=:red)
current_figure()