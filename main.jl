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
    "limit" => 300,
    "datasetKey" => "4fa7b334-ce0d-4e88-aaae-2e0c138d049e",
)
while length(presences) < min(count(presences), 2_000)
    occurrences!(presences)
end

presencelayer = mask(first(L), presences)
background = pseudoabsencemask(DistanceToEvent, presencelayer)
bgpoints = backgroundpoints(nodata(background, d -> d < 3), 2sum(presencelayer))

poly(CHE, color=:grey80)
scatter!(presencelayer, color=:red)
scatter!(bgpoints, color=:grey30, markersize=3)
lines!(CHE, color=:black)
current_figure()

# Set up the model
sdm = SDM(ZScore, Logistic, L, presencelayer, bgpoints)
hyperparameters!(classifier(sdm), :η, 5e-4);
hyperparameters!(classifier(sdm), :interactions, :all);
hyperparameters!(classifier(sdm), :epochs, 8_000);

# Train the model with optimal set of variables
variables!(sdm, ForwardSelection; included=[1], verbose=true)
#train!(sdm)

ConfusionMatrix(sdm) |> mcc

prd = predict(sdm, L; threshold=false)
heatmap(prd)
lines!(CHE, color=:black)
current_figure()

# VI
vi = variableimportance(sdm, [holdout(sdm)]; threshold=false)
miv = variables(sdm)[last(findmax(vi))]

scatter(features(sdm, 1), predict(sdm; threshold=false))
hlines!([threshold(sdm)], color=:red)
current_figure()

# Range
distrib = predict(sdm, L)

cs = cellsize(prd)

cmodel = deepcopy(sdm)
q = _estimate_q(cmodel, holdout(cmodel)...; α=0.005)

rlevels = LinRange(0.01, 0.2, 25)
qs = [_estimate_q(cmodel, holdout(cmodel)...; α=u) for u in rlevels]
scatter(rlevels, qs)

Cp, Ca = credibleclasses(prd, q)
heatmap(Ca .& Cp, colorrange=(0, 1))

# Partition
sure_presence = Cp .& (.!Ca)
heatmap(sure_presence)
sure_absence = Ca .& (.!Cp)
heatmap(sure_absence)
unsure = Ca .& Cp
heatmap(unsure)
unsure_in = unsure .& distrib
heatmap(unsure_in)
unsure_out = unsure .& (.!distrib)
heatmap(unsure_out)

poly(CHE, color=:grey90; axis=(; aspect=DataAspect()), figure=(; size=(1000, 550)))
heatmap!(nodata(sure_presence, false), colormap=[:transparent, :black, :black])
heatmap!(nodata(unsure_in, false), colormap=[:transparent, :forestgreen])
heatmap!(nodata(unsure_out, false), colormap=[:transparent, :orange])
lines!(CHE, color=:black)
contour!(distrib, color=:grey10, linewidth=1)
hidedecorations!(current_axis())
hidespines!(current_axis())
current_figure()
