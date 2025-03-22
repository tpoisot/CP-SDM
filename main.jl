using CairoMakie
include("lib.jl")

CHE = SpeciesDistributionToolkit.openstreetmap("Switzerland")
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
sp = taxon("Turdus torquatus")
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

poly(CHE)
scatter!(presencelayer)
scatter!(bgpoints)
current_figure()

# Set up the model
sdm = SDM(ZScore, Logistic, L, presencelayer, bgpoints)
hyperparameters!(classifier(sdm), :η, 1e-3);
hyperparameters!(classifier(sdm), :interactions, :all);
hyperparameters!(classifier(sdm), :epochs, 5_000);

# Train the model with optimal set of variables
variables!(sdm, ForwardSelection; included=[1], verbose=true)

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