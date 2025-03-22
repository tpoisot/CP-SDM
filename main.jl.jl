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
train!(sdm; training=St)

outputs = predict(sdm; threshold=false)[Sv]

conformalscore = zeros(length(outputs))
softmaxs = _softmax.(outputs)
for i in eachindex(conformalscore)
    conformalscore[i] = 1 - (labels(sdm)[Sv[i]] ? softmaxs[i][1] : softmaxs[i][2])
end

n = length(Sv)
α = 0.05
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

scatter(features(sdm, 1)[Sv], outputs, color=length.(classes))
hlines!([threshold(sdm)])
current_figure()