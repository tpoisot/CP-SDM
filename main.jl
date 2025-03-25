using CairoMakie
using SpeciesDistributionToolkit
using Statistics
import Downloads
import Dates
import PrettyTables

# Load the functions we need here
include("lib.jl")
include("cellsize.jl")
include("novelty.jl")
include("data.jl")

presencelayer = mask(first(L), Occurrences(records))
background = pseudoabsencemask(DistanceToEvent, presencelayer)
bgpoints = backgroundpoints(nodata(background, d -> d < 10), 3sum(presencelayer))

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
hyperparameters!(classifier(sdm), :epochs, 10_000);

# Train the model with optimal set of variables
variables!(sdm, ForwardSelection; verbose=true)

ConfusionMatrix(sdm) |> mcc

# Range
distrib = predict(sdm, L; threshold=true)

# Bootstrap
bsdm = Bagging(sdm, 50)
train!(bsdm)
bsvaria = predict(bsdm, L; threshold=false, consensus=iqr)

prd = predict(sdm, L; threshold=false)

f = Figure(; size=(600, 600))
ax = Axis(f[1,1]; aspect=DataAspect())
heatmap!(ax, prd, colormap=:navia, colorrange=(0,1))
for p in polygons
    lines!(ax, p, color=:grey50)
end
contour!(ax, distrib, color=:red, levels=1)
scatter!(ax, presencelayer, color=:white, strokecolor=:forestgreen, strokewidth=2)
hidespines!(ax)
hidedecorations!(ax)
current_figure()

# Uncertainty heatmap
f = Figure(; size=(600, 600))
ax = Axis(f[1,1]; aspect=DataAspect())
heatmap!(ax, quantize(bsvaria, 100), colormap=:nuuk, colorrange=(0,1))
for p in polygons
    lines!(ax, p, color=:grey50)
end
contour!(ax, distrib, color=:red, levels=1)
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

# Sensitivity analysis for the miscoverage rate
rlevels = LinRange(0.005, 0.15, 55)
qs = [_estimate_q(cmodel, holdout(cmodel)...; α=u) for u in rlevels]
surf_presence = zeros(length(qs))
surf_unsure = zeros(length(qs))
surf_unsure_presence = zeros(length(qs))
surf_unsure_absence = zeros(length(qs))
for i in eachindex(qs)
    Cp, Ca = credibleclasses(prd, qs[i])
    sure_presence = Cp .& (.!Ca)
    unsure = Ca .& Cp
    unsure_presence = unsure .& distrib
    unsure_absence = unsure .& (.!distrib)
    surf_presence[i] = sum(mask(cs, nodata(sure_presence, false)))
    surf_unsure[i] = sum(mask(cs, nodata(unsure, false)))
    surf_unsure_presence[i] = sum(mask(cs, nodata(unsure_presence, false)))
    surf_unsure_absence[i] = sum(mask(cs, nodata(unsure_absence, false)))
end

hlines([sum(mask(cs, nodata(distrib, false)))], color=:grey50, linestyle=:dash)
scatter!(rlevels, surf_presence .+ surf_unsure)
scatter!(rlevels, surf_presence)
current_figure()

q = median([_estimate_q(cmodel, fold...; α=0.05) for fold in kfold(cmodel; k=15)])
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
contour!(ax, distrib, color=:red, levels=1)
hidespines!(ax)
hidedecorations!(ax)
current_figure()

# This needs additional work: coverage by area of uncertainty
expl = explain(sdm, L; threshold=false)

mostdet = mosaic(x -> argmax(abs.(x)), expl)

shaplim(x) = maximum(abs.(quantile(x, [0.01, 0.99]))) .* (-1, 1)

# Important variables (Shapley) only on training data
svimp = [sum(abs.(ex)) for ex in expl]
smimp = last(findmax(svimp))

f = Figure(; size=(600, 600))
ax = Axis(f[1,1]; aspect=DataAspect())
heatmap!(ax, expl[smimp], colormap=:roma, colorrange=shaplim(expl[smimp]))
for p in polygons
    lines!(ax, p, color=:grey10, linewidth=1)
end
hidespines!(ax)
hidedecorations!(ax)
current_figure()

# Shapvals histogram
f = Figure()
ax_sa = Axis(f[1,1])
ax_sp = Axis(f[2,1])
ax_ua = Axis(f[1,2])
ax_up = Axis(f[2,2])
hist!(ax_sa, mask(expl[smimp], nodata(sure_absence, false)), bins=LinRange(-0.5, 0.5, 40), color=:red)
hist!(ax_sp, mask(expl[smimp], nodata(sure_presence, false)), bins=LinRange(-0.5, 0.5, 40), color=:green)
hist!(ax_ua, mask(expl[smimp], nodata(unsure_out, false)), bins=LinRange(-0.5, 0.5, 40), color=:pink)
hist!(ax_up, mask(expl[smimp], nodata(unsure_in, false)), bins=LinRange(-0.5, 0.5, 40), color=:lime)
for ax in [ax_sa, ax_sp, ax_ua, ax_up]
    xlims!(ax, (-0.5, 0.5))
    hideydecorations!(ax)
    tightlimits!(ax)
end
current_figure()

# Clim change
fprd = predict(sdm, F; threshold=false)
ft_distrib = predict(sdm, F; threshold=true)
nv = novelty(L, F, variables(sdm))

# Novelty map
f = Figure(; size=(600, 600))
ax = Axis(f[1,1]; aspect=DataAspect())
heatmap!(ax, (nv .- mean(nv))./std(nv), colormap=:broc, colorrange=shaplim((nv .- mean(nv))./std(nv)))
for p in polygons
    lines!(ax, p, color=:grey10, linewidth=1)
end
hidespines!(ax)
hidedecorations!(ax)
current_figure()

fCp, fCa = credibleclasses(fprd, q)

ft_sure_presence = fCp .& (.!fCa)
ft_sure_absence = fCa .& (.!fCp)
ft_unsure = fCa .& fCp
ft_unsure_in = ft_unsure .& ft_distrib
ft_unsure_out = ft_unsure .& (.!ft_distrib)

f = Figure(; size=(600, 600))
ax = Axis(f[1,1]; aspect=DataAspect())
for p in polygons
    poly!(ax, p, color=:grey90)
    lines!(ax, p, color=:grey10)
end
heatmap!(ax, nodata(ft_sure_presence, false), colormap=[:transparent, :black])
heatmap!(ax, nodata(ft_unsure_in, false), colormap=[:transparent, :forestgreen])
heatmap!(ax, nodata(ft_unsure_out, false), colormap=[:transparent, :orange])
contour!(ax, ft_distrib, color=:red, levels=1)
hidespines!(ax)
hidedecorations!(ax)
current_figure()

# Hists
f = Figure()
ax_sa = Axis(f[1,1]; xscale=identity)
ax_sp = Axis(f[2,1]; xscale=identity)
ax_ua = Axis(f[1,2]; xscale=identity)
ax_up = Axis(f[2,2]; xscale=identity)
hist!(ax_sa, mask(nv, nodata(ft_sure_absence, false)), bins=LinRange(0.1, 1.5, 40))
hist!(ax_sp, mask(nv, nodata(ft_sure_presence, false)), bins=LinRange(0.1, 1.5, 40))
hist!(ax_ua, mask(nv, nodata(ft_unsure_out, false)), bins=LinRange(0.1, 1.5, 40))
hist!(ax_up, mask(nv, nodata(ft_unsure_in, false)), bins=LinRange(0.1, 1.5, 40))
for ax in [ax_sa, ax_sp, ax_ua, ax_up]
    xlims!(ax, (0.1, 1.0))
    hideydecorations!(ax)
    tightlimits!(ax)
end
current_figure()

# Future scenarios transitions
f = Figure(; size=(600, 600))
ax = Axis(f[1,1]; aspect=DataAspect())
for p in polygons
    poly!(ax, p, color=:grey90)
    lines!(ax, p, color=:grey10)
end
heatmap!(ax, nodata(sure_presence .& ft_sure_presence,  false), colormap=[:black])
heatmap!(ax, nodata(sure_absence .& ft_unsure, false), colormap=[colorant"#6699ff"])
heatmap!(ax, nodata(sure_presence .& ft_unsure, false), colormap=[colorant"#ffcc66"])
heatmap!(ax, nodata(unsure .& ft_unsure, false), colormap=[:grey40])
heatmap!(ax, nodata((sure_absence .| unsure) .& ft_sure_presence, false), colormap=[colorant"#3333cc"]) # Certain gain
heatmap!(ax, nodata((sure_presence .| unsure) .& ft_sure_absence, false), colormap=[colorant"#ff6600"]) # Certain loss
for p in polygons
    lines!(ax, p, color=:grey10, linewidth=1)
end
hidespines!(ax)
hidedecorations!(ax)
Legend(
    f[2, 1],
    [PolyElement(; color = c) for c in [:black, :grey50, colorant"#ff6600", colorant"#ffcc66", colorant"#3333cc", colorant"#6699ff"]],
    ["Conserved", "Ambiguous", "Certain loss", "Possible loss", "Certain gain", "Possible gain"];
    orientation = :horizontal,
    nbanks = 2,
    framevisible = false,
    vertical = false,
)
current_figure()