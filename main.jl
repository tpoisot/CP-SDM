using CairoMakie
using SpeciesDistributionToolkit
using Statistics
import Downloads
import Dates
import PrettyTables

include("lib.jl")
include("novelty.jl")
include("data.jl")

presencelayer = mask(first(L), Occurrences(records))
background = pseudoabsencemask(DistanceToEvent, presencelayer)
bgpoints = backgroundpoints(nodata(background, d -> d < 10), 2sum(presencelayer))

f = Figure(; size=(600, 600))
ax = Axis(f[1,1]; aspect=DataAspect())
for p in polygons
    poly!(ax, p, color=:grey90)
    lines!(ax, p, color=:grey10)
end
scatter!(ax, presencelayer, color=:white, strokecolor=:forestgreen, strokewidth=2)
scatter!(ax, bgpoints, color=:grey30, markersize=4)
hidespines!(ax)
hidedecorations!(ax)
current_figure()

# Set up the model
sdm = SDM(ZScore, Logistic, L, presencelayer, bgpoints)
hyperparameters!(classifier(sdm), :η, 1e-3);
hyperparameters!(classifier(sdm), :interactions, :all);
hyperparameters!(classifier(sdm), :epochs, 8_000);

# Train the model with optimal set of variables
variables!(sdm, ForwardSelection; verbose=true)

ConfusionMatrix(sdm) |> mcc

# Range
distrib = predict(sdm, L; threshold=true)
#bsvaria = predict(sdm, L; threshold=false, consensus=iqr)
prd = predict(sdm, L; threshold=false)

f = Figure(; size=(600, 600))
ax = Axis(f[1,1]; aspect=DataAspect())
heatmap!(ax, prd, colormap=:tempo, colorrange=(0,1))
for p in polygons
    lines!(ax, p, color=:grey50)
end
contour!(ax, distrib, color=:red, levels=1, linestyle=:dot)
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
q = median([_estimate_q(cmodel, fold...; α=0.07) for fold in kfold(cmodel; k=15)])

# rlevels = LinRange(0.01, 0.2, 25)
# qs = [_estimate_q(cmodel, holdout(cmodel)...; α=u) for u in rlevels]
# scatter(rlevels, qs)

Cp, Ca = credibleclasses(prd, q)
heatmap(Ca .& Cp, colorrange=(0, 1))

# Partition
sure_presence = Cp .& (.!Ca)
sure_absence = Ca .& (.!Cp)
unsure = Ca .& Cp
unsure_in = unsure .& distrib
unsure_out = unsure .& (.!distrib)

f = Figure(; size=(600, 600))
ax = Axis(f[1,1]; aspect=DataAspect())
for p in polygons
    poly!(ax, p, color=:grey90)
    lines!(ax, p, color=:grey10)
end
heatmap!(ax, nodata(sure_presence, false), colormap=[:transparent, :black])
heatmap!(ax, nodata(unsure_in, false), colormap=[:transparent, :forestgreen])
heatmap!(ax, nodata(unsure_out, false), colormap=[:transparent, :orange])
contour!(ax, distrib, color=:red, levels=1, linestyle=:dot)
scatter!(ax, presencelayer, color=:white, strokecolor=:forestgreen, strokewidth=2)
hidespines!(ax)
hidedecorations!(ax)
current_figure()

# This needs additional work: coverage by area of uncertainty
expl = explain(sdm, L; threshold=false)

mostdet = mosaic(x -> argmax(abs.(x)), expl)

shaplim(x) = maximum(abs.(quantile(x, [0.05, 0.95]))) .* (-1, 1)

# Important variables (Shapley) only on training data
svimp = [sum(abs.(ex)) for ex in expl]
smimp = last(findmax(svimp))

heatmap(expl[smimp], colormap=:curl, colorrange=shaplim(expl[smimp]))
for p in polygons
    lines!(p, color=:grey10, linewidth=1)
end
current_figure()

# Clim change
fprd = predict(sdm, F; threshold=false)
nv = novelty(L, F, varaibles(sdm))

fCp, fCa = credibleclasses(fprd, q)

ft_sure_presence = fCp .& (.!fCa)
ft_sure_absence = fCa .& (.!fCp)
ft_unsure = fCa .& fCp

f = Figure(; size=(600, 600))
ax = Axis(f[1,1]; aspect=DataAspect())
for p in polygons
    poly!(ax, p, color=:grey90)
    lines!(ax, p, color=:grey10)
end
heatmap!(ax, nodata(sure_presence, false), colormap=[:transparent, :black])
heatmap!(ax, nodata(unsure, false), colormap=[:transparent, :forestgreen])
hidespines!(ax)
hidedecorations!(ax)
current_figure()
