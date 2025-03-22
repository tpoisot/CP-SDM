using CairoMakie
using SpeciesDistributionToolkit
using Statistics
import Downloads
import Dates

include("lib.jl")

include("data.jl")

presencelayer = mask(first(L), Occurrences(records))
background = pseudoabsencemask(DistanceToEvent, presencelayer)
bgpoints = backgroundpoints(nodata(background, d -> d < 10), 2sum(presencelayer))

f = Figure()
ax = Axis(f[1,1]; aspect=DataAspect())
for p in polygons
    poly!(ax, p, color=:grey90)
    lines!(ax, p, color=:grey10)
end
scatter!(ax, presencelayer, color=:white, strokecolor=:forestgreen, strokewidth=2)
scatter!(ax, bgpoints, color=:grey30, markersize=4)
current_figure()

# Set up the model
sdm = Bagging(SDM(PCATransform, DecisionTree, L, presencelayer, bgpoints), 25)
#hyperparameters!(classifier(sdm), :η, 1e-3);
#hyperparameters!(classifier(sdm), :interactions, :self);
#hyperparameters!(classifier(sdm), :epochs, 8_000);

# Train the model with optimal set of variables
variables!(sdm, ForwardSelection; verbose=true, bagfeatures=true)

ConfusionMatrix(sdm) |> mcc

# Range
distrib = predict(sdm, L; threshold=true, consensus=majority)
prd = predict(sdm, L; threshold=false)

f = Figure()
ax = Axis(f[1,1]; aspect=DataAspect())
heatmap!(ax, prd, colormap=:tempo, colorrange=(0,1))
for p in polygons
    lines!(ax, p, color=:grey10)
end
scatter!(ax, presencelayer, color=:white, strokecolor=:forestgreen, strokewidth=2)
hidespines!(ax)
hidedecorations!(ax)
current_figure()

# VI
vi = variableimportance(sdm, [holdout(sdm)]; threshold=false)
miv = variables(sdm)[last(findmax(vi))]

scatter(features(sdm, miv), predict(sdm; threshold=false))
hlines!([threshold(sdm)], color=:red)
current_figure()

cs = cellsize(prd)

cmodel = deepcopy(sdm)
q = median([_estimate_q(cmodel, fold...; α=0.1) for fold in kfold(cmodel; k=15)])

# rlevels = LinRange(0.01, 0.2, 25)
# qs = [_estimate_q(cmodel, holdout(cmodel)...; α=u) for u in rlevels]
# scatter(rlevels, qs)

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


f = Figure()
ax = Axis(f[1,1]; aspect=DataAspect())
for p in polygons
    poly!(ax, p, color=:grey90)
    lines!(ax, p, color=:grey10)
end
heatmap!(ax, nodata(sure_presence, false), colormap=[:transparent, :black])
heatmap!(ax, nodata(unsure_in, false), colormap=[:transparent, :forestgreen])
heatmap!(ax, nodata(unsure_out, false), colormap=[:transparent, :orange])
scatter!(ax, presencelayer, color=:white, strokecolor=:forestgreen, strokewidth=2)
hidespines!(ax)
hidedecorations!(ax)
current_figure()
