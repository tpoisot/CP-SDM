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
    s = _softmax.(f̂)
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
        s₊, s₋ = _softmax(prediction[k])
        if s₊ >= (1-q)
            presence[k] = true
        end
        if s₋ >= (1-q)
            absence[k] = true
        end
    end
    return (presence, absence)
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
while length(presences) < min(count(presences), 1_000)
    occurrences!(presences)
end

presencelayer = mask(first(L), presences)
background = pseudoabsencemask(DistanceToEvent, presencelayer)
bgpoints = backgroundpoints(nodata(background, d -> d < 2), 3sum(presencelayer))

sdm = SDM(ZScore, Logistic, L, presencelayer, bgpoints)
variables!(sdm, ForwardSelection; verbose=true)
hyperparameters!(classifier(sdm), :η, 1e-3);
hyperparameters!(classifier(sdm), :interactions, :all);
hyperparameters!(classifier(sdm), :epochs, 5_000);

train!(sdm)

ConfusionMatrix(sdm) |> mcc

prd = predict(sdm, L; threshold=false)
heatmap(prd)

cmodel = deepcopy(sdm)
q = _estimate_q(cmodel, holdout(cmodel)...; α=0.1)

Cp, Ca = credibleclasses(prd, q)
heatmap(Ca .& Cp, colorrange=(0, 1))

poly(CHE, color=:white, strokecolor=:black, strokewidth=1)
heatmap!(nodata(Cp, false), colormap=[:transparent, :grey80])
heatmap!(nodata(Ca .& Cp, false), colormap=[:transparent, :grey20])
scatter!(mask(presences, CHE), markersize=3, color=:orange)
current_figure()