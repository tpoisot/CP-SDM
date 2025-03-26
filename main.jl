using CairoMakie
using SpeciesDistributionToolkit
using Statistics
import Downloads
import Dates
import PrettyTables
import Random
Random.seed!(42069)

# Load the functions we need here
include("theme.jl")
include("lib.jl")
include("cellsize.jl")
include("novelty.jl")
include("data.jl")

# Paths to store outputs
fpath = joinpath(@__DIR__, "figures")
apath = joinpath(@__DIR__, "artifacts")
if ~ispath(fpath)
    mkpath(fpath)
end
if ~ispath(apath)
    mkpath(apath)
end

# Generate pseudo-absences
δ = 0.7 # Tapering for distance
presencelayer = mask(first(L), Occurrences(records))
background = pseudoabsencemask(DistanceToEvent, presencelayer)
bgpoints = backgroundpoints(nodata(background, d -> d < 10).^δ, 3sum(presencelayer))

# Plot the figure for presence/absence
f = Figure(; size=(600, 600))
ax = Axis(f[1,1]; aspect=DataAspect())
for p in polygons
    poly!(ax, p, color=:grey95)
    lines!(ax, p, color=:grey10)
end
scatter!(ax, presencelayer, color=:white, strokecolor=:forestgreen, strokewidth=2)
scatter!(ax, bgpoints, color=:grey30, markersize=4)
hidespines!(ax)
hidedecorations!(ax)
CairoMakie.save(joinpath(fpath, "occurrrences.png"), current_figure())
current_figure()

# Set up the model - logistic regression with Z-score before
sdm = SDM(ZScore, Logistic, L, presencelayer, bgpoints)
hyperparameters!(classifier(sdm), :η, 1e-3) # Slow descent
hyperparameters!(classifier(sdm), :interactions, :all) # All interactions
hyperparameters!(classifier(sdm), :epochs, 10_000) # Longer training

# Train the model with optimal set of variables, using forward selection and MCC
# as the measure
variables!(sdm, ForwardSelection; verbose=true)

# Measure of model performance
ConfusionMatrix(sdm) |> mcc

# Range
distrib = predict(sdm, L; threshold=true)

# Bootstrap to get to uncertainty - we re-train 50 models with the same
# features, but different bags
bsdm = Bagging(sdm, 50)
bsdm |> outofbag |> (M) -> 1 - accuracy(M) # OOB error
train!(bsdm)
bsvaria = predict(bsdm, L; threshold=false, consensus=iqr)

# Prediction based on baseline data
prd = predict(sdm, L; threshold=false)

f = Figure(; size=(1200, 600))
ax1 = Axis(f[1,1]; aspect=DataAspect())
hm1 = heatmap!(ax1, prd, colormap=:navia, colorrange=(0,1))
Colorbar(
    f[1, 1],
    hm1;
    label = "Prediction (presence)",
    alignmode = Inside(),
    height = Relative(0.5),
    flipaxis = false,
    valign = :bottom,
    halign = :right,
    tellheight = false,
    tellwidth = false,
    vertical = true,
)
ax2 = Axis(f[1,2]; aspect=DataAspect())
hm2 = heatmap!(ax2, quantize(bsvaria, 100), colormap=:nuuk, colorrange=(0,1))
Colorbar(
    f[1, 2],
    hm2;
    label = "Inter-quantile range",
    alignmode = Inside(),
    height = Relative(0.5),
    flipaxis = false,
    valign = :bottom,
    halign = :right,
    tellheight = false,
    tellwidth = false,
    vertical = true,
)
for ax in [ax1, ax2]
    hidespines!(ax)
    hidedecorations!(ax)
    for p in polygons
        lines!(ax, p, color=:grey50)
    end
    contour!(ax, distrib, color=:red, levels=1)
end
CairoMakie.save(joinpath(fpath, "prediction.png"), current_figure())
current_figure()

# VI
vi = variableimportance(sdm, kfold(sdm); threshold=false)
miv = variables(sdm)[last(findmax(vi))]

scatter(features(sdm, miv), predict(sdm; threshold=false))
hlines!([threshold(sdm)], color=:red)
current_figure()

cs = cellsize(prd)

cmodel = deepcopy(sdm)

# Sensitivity analysis for the miscoverage rate
rlevels = LinRange(0.01, 0.2, 50)
qs = [_estimate_q(cmodel, holdout(cmodel)...; α=u) for u in rlevels]
surf_presence = zeros(length(qs))
surf_unsure = zeros(length(qs))
surf_unsure_presence = zeros(length(qs))
surf_unsure_absence = zeros(length(qs))

𝐏 = predict(sdm; threshold=false)
eff = [mean(length.(credibleclasses.(𝐏, q))) for q in qs]

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

# Cross-conformal with median range selected
q = median([_estimate_q(cmodel, fold...; α=0.05) for fold in kfold(cmodel; k=10)])
Cp, Ca = credibleclasses(prd, q)

# Partition
sure_presence = Cp .& (.!Ca)
sure_absence = Ca .& (.!Cp)
unsure = Ca .& Cp
unsure_in = unsure .& distrib
unsure_out = unsure .& (.!distrib)

# Big figure with range and uncertainty
f = Figure(; size=(1200, 600))
ax = Axis(f[1:2,1]; aspect=DataAspect())
for p in polygons
    poly!(ax, p, color=:grey95)
    lines!(ax, p, color=:grey10)
end
heatmap!(ax, nodata(sure_presence, false), colormap=[:transparent, :black])
heatmap!(ax, nodata(unsure_in, false), colormap=[:transparent, :forestgreen])
heatmap!(ax, nodata(unsure_out, false), colormap=[:transparent, :orange])
contour!(ax, distrib, color=:red, levels=1)
hidespines!(ax)
hidedecorations!(ax)
ax2 = Axis(f[1,2], xlabel="Risk level α", yscale=log10, ylabel="Range (km²)")
ylims!(ax2, 1e4, 1e6)
hlines!(ax2, [sum(mask(cs, nodata(distrib, false)))], color=:grey50, linestyle=:dash, label="SDM range")
scatter!(ax2, rlevels, surf_presence, color=:grey10, marker=:rect, label="Sure range")
scatter!(ax2, rlevels, surf_presence .+ surf_unsure, color=:grey40, label="Total range")
axislegend(ax2, position=:rb)
ax3 = Axis(f[2,2], xlabel="Inter-quantile range (ensemble)")
# Bins
bins = LinRange(0.0, round(quantile(bsvaria, 0.9); digits=2), 80)
hiparams = (; bins=bins, normalization=:pdf)
hist!(ax3, mask(bsvaria, nodata(sure_absence, false)); color=(:orange, 0.7), label="Sure absence", hiparams...)
hist!(ax3, mask(bsvaria, nodata(unsure, false)); color=(:grey80, 0.7), label="Unsure", hiparams...)
hist!(ax3, mask(bsvaria, nodata(sure_presence, false)); color=(:forestgreen, 0.7), label="Sure presence", hiparams...)
axislegend(ax3, nbanks=3)
tightlimits!(ax3)
hideydecorations!(ax3)
hidespines!(ax3, :r)
hidespines!(ax3, :l)
hidespines!(ax3, :t)
CairoMakie.save(joinpath(fpath, "conformalrange.png"), current_figure())
current_figure()

# Example with unknown areas
q2 = median([_estimate_q(cmodel, fold...; α=0.2) for fold in kfold(cmodel; k=10)])
Cp2, Ca2 = credibleclasses(prd, q2)
undet = .!(Cp2 .| Ca2)

# Big figure with range and uncertainty
f = Figure(; size=(1200, 600))
ax = Axis(f[1:2,1]; aspect=DataAspect())
for p in polygons
    poly!(ax, p, color=:grey95)
    lines!(ax, p, color=:grey10)
end
heatmap!(ax, nodata(undet, false), colormap=[:transparent, :red])
contour!(ax, distrib, color=:red, levels=1)
hidespines!(ax)
hidedecorations!(ax)
ax2 = Axis(f[1,2], xlabel="Risk level α", ylabel="Efficiency")
hlines!(ax2, [1.0], color=:grey50, linestyle=:dash)
scatter!(ax2, rlevels, eff, color=:black)
CairoMakie.save(joinpath(fpath, "undetrange.png"), current_figure())
current_figure()

# Shapley values
S = explain(sdm, L; threshold=false)

# Most important Shapley value (for fun, not used in the paper)
mostdet = mosaic(x -> argmax(abs.(x)), S)

# Custom function for Shapley limits
shaplim(x) = maximum(abs.(quantile(x, [0.13, 0.87]))) .* (-1, 1)

# Important variables (Shapley) only on training data
svimp = [mean(abs.(ex)) for ex in S]
smimp = last(findmax(svimp))

# Map with Shapley and contributions
f = Figure(; size=(1200, 600))
ax = Axis(f[1:2,1]; aspect=DataAspect())
hm = heatmap!(ax, S[smimp], colormap=:RdYlBu, colorrange=shaplim(S[smimp]))
for p in polygons
    lines!(ax, p, color=:grey10, linewidth=1)
end
Colorbar(
    f[1:2, 1],
    hm;
    label = "Impact on prediction",
    alignmode = Inside(),
    height = Relative(0.4),
    flipaxis = false,
    valign = :bottom,
    halign = :right,
    tellheight = false,
    tellwidth = false,
    vertical = true,
)
hidespines!(ax)
hidedecorations!(ax)
ax2 = Axis(f[1,2], xlabel="Effect on prediction")
bins = LinRange(shaplim(S[smimp])..., 100)
hist!(ax2, mask(S[smimp], nodata(sure_absence, false)), color=(:orange, 0.7), bins=bins, normalization=:pdf)
hist!(ax2, mask(S[smimp], nodata(sure_presence, false)), color=(:grey80, 0.7), bins=bins, normalization=:pdf)
hist!(ax2, mask(S[smimp], nodata(unsure, false)), color=(:forestgreen, 0.7), bins=bins, normalization=:pdf)
tightlimits!(ax2)
hideydecorations!(ax2)
hidespines!(ax2, :r)
hidespines!(ax2, :l)
hidespines!(ax2, :t)
ax3 = Axis(f[2,2], xlabel="Variable", ylabel="Relative importance")
surea_imp = [mean(abs.(mask(ex, nodata(sure_absence, false)))) for ex in S]
uns_imp = [mean(abs.(mask(ex, nodata(unsure, false)))) for ex in S]
surep_imp = [mean(abs.(mask(ex, nodata(sure_presence, false)))) for ex in S]
scatterlines!(ax3, svimp./sum(svimp), color=:black, linewidth=1, linestyle=:dot)
scatterlines!(ax3, surea_imp./sum(surea_imp), color=:orange, label="Sure absence")
scatterlines!(ax3, uns_imp./sum(uns_imp), color=:grey60, label="Unsure")
scatterlines!(ax3, surep_imp./sum(surep_imp), color=:forestgreen, label="Sure presence")
axislegend(ax3)
CairoMakie.save(joinpath(fpath, "shapley.png"), current_figure())
current_figure()

# Clim change
fprd = predict(sdm, F; threshold=false)
ft_distrib = predict(sdm, F; threshold=true)
nv = novelty(L, F, variables(sdm))

# Novelty map
f = Figure(; size=(600, 600))
ax = Axis(f[1,1]; aspect=DataAspect())
hm = heatmap!(ax, (nv .- mean(nv))./std(nv), colormap=:broc, colorrange=shaplim((nv .- mean(nv))./std(nv)))
for p in polygons
    lines!(ax, p, color=:grey10, linewidth=1)
end
Colorbar(
    f[1, 1],
    hm;
    label = "Relative climate novelty",
    alignmode = Inside(),
    height = Relative(0.4),
    flipaxis = false,
    valign = :bottom,
    halign = :right,
    tellheight = false,
    tellwidth = false,
    vertical = true,
)
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

cmap = [colorant"#fdb863", colorant"#e66101", colorant"#020202",colorant"#5e3c99", colorant"#b2abd2"]

# Future scenarios transitions
f = Figure(; size=(600, 600))
ax = Axis(f[1,1]; aspect=DataAspect())
for p in polygons
    poly!(ax, p, color=:grey95)
    lines!(ax, p, color=:grey10)
end
heatmap!(ax, nodata(sure_presence .& ft_sure_presence,  false), colormap=[cmap[3]])
heatmap!(ax, nodata(sure_absence .& ft_unsure, false), colormap=[cmap[5]])
heatmap!(ax, nodata(sure_presence .& ft_unsure, false), colormap=[cmap[1]])
heatmap!(ax, nodata(unsure .& ft_unsure, false), colormap=[:grey70])
heatmap!(ax, nodata((sure_absence .| unsure) .& ft_sure_presence, false), colormap=[cmap[4]]) # Certain gain
heatmap!(ax, nodata((sure_presence .| unsure) .& ft_sure_absence, false), colormap=[cmap[2]]) # Certain loss
for p in polygons
    lines!(ax, p, color=:grey10, linewidth=1)
end
hidespines!(ax)
hidedecorations!(ax)
Legend(
    f[1, 1],
    alignmode = Inside(),
    height = Relative(0.4),
    valign = :bottom,
    halign = :right,
    tellheight = false,
    tellwidth = false,
    [PolyElement(; color = c) for c in [cmap..., :grey70]],
    ["Possible loss", "Sure loss", "Conserved", "Sure gain", "Possible gain", "Ambiguous"];
    orientation = :vertical,
    nbanks = 1,
    framevisible = false,
    vertical = false,
)
current_figure()