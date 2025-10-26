using CairoMakie
using SpeciesDistributionToolkit
const SDT = SpeciesDistributionToolkit
using Statistics
import Dates
import PrettyTables
import Random
using ColorBlendModes
using ColorSchemes
Random.seed!(42069)

# Paths to store outputs
fpath = joinpath(@__DIR__, "figures")
apath = joinpath(@__DIR__, "artifacts")
if ~ispath(fpath)
    mkpath(fpath)
end
if ~ispath(apath)
    mkpath(apath)
end

# Basic data
gadm_usa_level1 = getpolygon(PolygonData(GADM, Countries); level=1, country="USA")
polygons = [
    gadm_usa_level1["California"],
    gadm_usa_level1["Idaho"],
    gadm_usa_level1["Nevada"],
    gadm_usa_level1["Oregon"],
    gadm_usa_level1["Washington"],
]
landmass = FeatureCollection(polygons)
records = mask(OccurrencesInterface.__demodata(), landmass);

# Load the functions we need here
include("utils/theme.jl")
include("utils/conformal.jl")
include("utils/novelty.jl")
#include("utils/data.jl")

# Map of occurrences
f = Figure()
ax = Axis(f[1,1])
for p in polygons
    lines!(ax, p, color=:black)
end
scatter!(ax, records; markersize=6, color=:orange)
#scatter!(ax, bgpoints; markersize=5, color=:grey50)
current_figure()

# Set up the model - logistic regression with Z-score before
sdm = SDeMo.loadsdm("artifacts/sdm.json")
threshold!(sdm)
folds = kfold(sdm)

# VI
vi = variableimportance(sdm, folds; threshold=false)
miv = variables(sdm)[last(findmax(vi))]

renderfigure("occurrences")

# Measure of model performance
ConfusionMatrix(sdm) |> mcc

cv = crossvalidate(sdm, folds)
map(ppv, cv)
map(npv, cv)
map(mcc, cv)
map(κ, cv)
map(trueskill, cv)
map(accuracy, cv)

# Bootstrap to get to uncertainty - we re-train 50 models with the same
# features, but different bags
bsdm = Bagging(sdm, 50)
bsdm |> outofbag |> accuracy # OOB error
train!(bsdm)

# Load the environmental variables
L = [SimpleSDMLayers._read_geotiff("artifacts/historical.tif"; bandnumber=i) for i in 1:19]
bootstrap_variability = predict(bsdm, L; threshold=false, consensus=iqr)

# Prediction based on baseline data
current_range = predict(sdm, L; threshold=true)
current_score = predict(sdm, L; threshold=false)

renderfigure("prediction")

# this is where the experiments start
function q(sdm; α=0.05, folds=kfold(sdm))
    mc_q = [conformal(sdm, f...; α=α) for f in folds]
    return vec(median(vcat([hcat(q...) for q in mc_q]...), dims= 1))
end

q₊, q₋ = q(sdm)

Ŷ = predict(sdm, L; threshold=false)
uncertain = (x -> length(credibleclasses(x, q₊, q₋))).(Ŷ)
heatmap(predict(sdm, L), colormap=[:white, :darkgreen])
heatmap!(uncertain, colormap=[:transparent, :grey80])
contour!(predict(sdm, L), color=:darkgreen)
lines!(landmass, color=:black)
scatter!(records, color=:purple, markersize=4)
current_figure()
# normal code resumes

cell_surface = cellarea(current_range)

cmodel = deepcopy(sdm)

# Sensitivity analysis for the miscoverage rate
risk_levels = repeat(LinRange(0.7, 0.99, 15); inner=10)
qs = [conformal(cmodel, holdout(cmodel)...; α=1.0-risk) for risk in risk_levels]

function _agr(rl, qs)
    x = sort(unique(rl))
    ym = zeros(length(x))
    ys = zeros(length(x))
    for i in eachindex(x)
        yi = qs[findall(rl .== x[i])]
        ym[i] = mean(yi)
        ys[i] = std(yi)#1.96 * std(yi)/length(yi)
    end
    return x, ym, ys
end

x, m, s = _agr(risk_levels, first.(qs))
errorbars(x, m, s, whiskerwidth=10, color=:darkgreen, depth_shift=-1.0)
scatter!(x, m, strokecolor=:darkgreen, strokewidth=3, color=:white, label="Presence.")
x, m, s = _agr(risk_levels, last.(qs))
errorbars!(x, m, s, whiskerwidth=10, color=:grey50, depth_shift=-1.0)
scatter!(x, m, strokecolor=:grey50, strokewidth=3, color=:white, label="Absence.")
axislegend(position=:rb)
current_figure()

# Risk level at which an area becomes certain
iscertain(p, q1, q2) = length(credibleclasses(p, q1, q2)) == 1
uncmap = [(y -> iscertain(y, q...)).(predict(sdm, L; threshold=false)) for q in qs]
function uncindex(v, rl)
    u = findall(v)
    return isempty(u) ? NaN : maximum(rl[u])
end
uncmosaic = mosaic(v -> uncindex(v, risk_levels), uncmap)
fg, ax, hm = heatmap(uncmosaic, colormap=:davos)
lines!(ax, landmass, color=:black)
Colorbar(fg[1,2], hm)
current_figure()

# Now we do the uncertainty figure

surf_presence = zeros(length(qs))
surf_undet = zeros(length(qs))
surf_unsure = zeros(length(qs))
surf_unsure_presence = zeros(length(qs))
surf_unsure_absence = zeros(length(qs))

𝐏 = predict(sdm; threshold=false)
eff = [mean(length.(credibleclasses.(𝐏, q...))) for q in qs]

scatter(risk_levels, eff)
errorbars(_agr(risk_levels, eff)...)

for i in eachindex(qs)
    Cp, Ca = credibleclasses(current_score, qs[i]...)
    undet = .!(Cp .| Ca)
    sure_presence = Cp .& (.!Ca)
    unsure = Ca .& Cp
    unsure_presence = unsure .& current_range
    unsure_absence = unsure .& (.!current_range)
    surf_presence[i] = sum(mask(cell_surface, nodata(sure_presence, false)))
    surf_undet[i] = sum(mask(cell_surface, nodata(undet, false)))
    surf_unsure[i] = sum(mask(cell_surface, nodata(unsure, false)))
    surf_unsure_presence[i] = sum(mask(cell_surface, nodata(unsure_presence, false)))
    surf_unsure_absence[i] = sum(mask(cell_surface, nodata(unsure_absence, false)))
end

# Cross-conformal with median range selected
Cp, Ca = credibleclasses(current_score, q₊, q₋)

# Partition
sure_presence = Cp .& (.!Ca)
sure_absence = Ca .& (.!Cp)
unsure = Ca .& Cp
unsure_in = unsure .& current_range
unsure_out = unsure .& (.!current_range)

renderfigure("uncertainty")

# Example with unknown areas
uq = q(sdm; α=0.2)
uCp, uCa = credibleclasses(current_score, uq...)
undet = .!(uCp .| uCa)

renderfigure("undetrange")

# Shapley values
S = explain(sdm, L; threshold=false, samples=100)

# Most important Shapley value (for fun, not used in the paper)
mostdet = mosaic(x -> argmax(abs.(x)), S)

# Custom function for Shapley limits
shaplim(x) = maximum(abs.(quantile(x, [0.13, 0.87]))) .* (-1, 1)

# Important variables (Shapley) only on training data
svimp = [mean(abs.(ex)) for ex in S]
smimp = last(findmax(svimp))

renderfigure("shapley")

# Clim change
projected_predictions = [predict(sdm, v; threshold=false) for (k, v) in F]
projected_ranges = [predict(sdm, v; threshold=true) for (k, v) in F]
projected_prediction = mosaic(median, projected_predictions)

heatmap(projected_prediction)
contour!(projected_range)
current_figure()

projected_range = mosaic(majority, projected_ranges)

# Conformal CP
mc_q = [conformal(sdm, f...; α=0.05) for f in kfold(sdm)]
q = vec(median(vcat([hcat(q...) for q in mc_q]...), dims= 1))
Cp, Ca = credibleclasses(prd, q₊, q₋)

fCp, fCa = credibleclasses(projected_prediction, q...)

ft_sure_presence = fCp .& (.!fCa)
ft_sure_absence = fCa .& (.!fCp)
ft_unsure = fCa .& fCp
ft_unsure_in = ft_unsure .& projected_range
ft_unsure_out = ft_unsure .& (.!projected_range)

renderfigure("gainloss")

# Code for novelty - we take the median value for each pixel here purely to save
# time, otherwise this is a very long step for very minor differences in the end

Fmed = [mosaic(median, [F[m][i] for m in keys(F)]) for i in eachindex(L)]

novel_climates = novelty(L, Fmed, variables(sdm)) # Emergence of novel climates
lost_climates = novelty(Fmed, L, variables(sdm)) # Loss of historical climates

renderfigure("novelty")