using SpeciesDistributionToolkit
using CairoMakie
using Statistics

X, y = SDeMo.__demodata()

model = SDM(ZScore, NaiveBayes, X, y)
variables!(model, ForwardSelection; included=[1,12], verbose=true)

ConfusionMatrix(model) |> mcc

# Keep a calibration set
St, Sv = holdout(model)
train!(model, training=St)

outputs = predict(model; threshold=false)[Sv]

function _softmax(p)
    w = [exp(p), exp(1-p)]
    return w ./ sum(w)
end

function _no_softmax(p)
    w = [p, 1-p]
    return w
end

conformalscore = zeros(length(outputs))
softmaxs = _softmax.(outputs)
for i in eachindex(conformalscore)
    conformalscore[i] = 1 - (labels(model)[Sv[i]] ? softmaxs[i][1] : softmaxs[i][2])
end

n = length(Sv)
α = 0.1
qi = (n+1)*(1-α)/n
q = quantile(conformalscore, qi)

classes = Vector{Set{Bool}}(undef, length(Sv))
for i in eachindex(classes)
    s = Bool[]
    f_true, f_false = _softmax(outputs[i])
    if f_true >= (1-q)
        push!(s, true)
    end
    if f_false >= (1-q)
        push!(s, false)
    end
    classes[i] = Set(s)
end


# Step 2 

CHE = SpeciesDistributionToolkit.openstreetmap("Switzerland")
bio_vars = [1, 11, 5, 8, 6]
provider = RasterData(CHELSA2, BioClim)
L = SDMLayer{Float32}[
    SDMLayer(
        provider;
        layer = x,
        SpeciesDistributionToolkit.boundingbox(CHE)...,
    ) for x in bio_vars
];
mask!(L, CHE)
ouzel = taxon("Turdus torquatus")
presences = occurrences(
    ouzel,
    first(L),
    "occurrenceStatus" => "PRESENT",
    "limit" => 300,
    "datasetKey" => "4fa7b334-ce0d-4e88-aaae-2e0c138d049e",
)
while length(presences) < count(presences)
    occurrences!(presences)
end

presencelayer = mask(first(L), Occurrences(mask(presences, CHE)))
background = pseudoabsencemask(DistanceToEvent, presencelayer)
bgpoints = backgroundpoints(nodata(background, d -> d < 4), 2sum(presencelayer))

sdm = SDM(ZScore, Logistic, L, presencelayer, bgpoints)
hyperparameters!(classifier(sdm), :η, 1e-3);
hyperparameters!(classifier(sdm), :interactions, :all);
hyperparameters!(classifier(sdm), :epochs, 5_000);

St, Sv = holdout(sdm)

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

cmodel = deepcopy(sdm)
q = _estimate_q(cmodel, holdout(cmodel)...; α=0.05)

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

prd = predict(sdm, L; threshold=false)

Cp, Ca = credibleclasses(prd, q)
heatmap(Cp .+ Ca)