using CairoMakie
using SpeciesDistributionToolkit
const SDT = SpeciesDistributionToolkit
using Statistics
import Downloads
import Dates
import PrettyTables
import Random
Random.seed!(42069)

# Load the functions we need here
include("utils/theme.jl")
include("utils/conformal.jl")
include("utils/novelty.jl")
include("utils/data.jl")

# TODO figure of the q+ and q- when changing the coverage threshold to demo Mondrian

# cp = [credible(y, (q₊, q₋)) for y in predict(model; threshold=false)]
# 
# uncertain = (x -> length(credible(x, (q₊, q₋)))).(U)
# heatmap(predict(model, L), colormap=[:white, :darkgreen])
# heatmap!(uncertain, colormap=[:transparent, :grey70])
# contour!(predict(model, L), color=:darkgreen)
# lines!(landmass, color=:black)
# scatter!(presencelayer, color=:lime, markersize=4)
# current_figure()

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
presencelayer = mask(first(L), records)
background = pseudoabsencemask(DistanceToEvent, presencelayer)
bgpoints = backgroundpoints(nodata(background, d -> !(20 <= d <= 300)), 3sum(presencelayer))

# Map of occurrences
f = Figure()
ax = Axis(f[1,1])
for p in polygons
    lines!(ax, p, color=:black)
end
scatter!(ax, presencelayer; markersize=6, color=:orange)
scatter!(ax, bgpoints; markersize=5, color=:grey50)
current_figure()

# Set up the model - logistic regression with Z-score before
sdm = SDM(ZScore, Logistic, L, presencelayer, bgpoints)
hyperparameters!(classifier(sdm), :η, 1e-3) # Slow descent
hyperparameters!(classifier(sdm), :interactions, :all) # All interactions
hyperparameters!(classifier(sdm), :epochs, 10000) # Longer training

# Folds
folds = kfold(sdm)

# Train the model with optimal set of variables, using forward selection and MCC
# as the measure
variables!(sdm, ForwardSelection, folds; verbose=true)
threshold!(sdm)

# VI
vi = variableimportance(sdm, folds; threshold=false)
miv = variables(sdm)[last(findmax(vi))]

renderfigure("occurrences")

# Measure of model performance
# Make a PrettyTable for output
ConfusionMatrix(sdm) |> mcc

cv = crossvalidate(sdm, folds)
map(ppv, cv)
map(npv, cv)
map(mcc, cv)
map(κ, cv)
map(trueskill, cv)
map(accuracy, cv)

# Range
distrib = predict(sdm, L; threshold=true)

# Bootstrap to get to uncertainty - we re-train 50 models with the same
# features, but different bags
bsdm = Bagging(sdm, 50)
bsdm |> outofbag |> accuracy # OOB error
train!(bsdm)
bsvaria = predict(bsdm, L; threshold=false, consensus=iqr)

# Prediction based on baseline data
prd = predict(sdm, L; threshold=false)

renderfigure("prediction")

# this is where the experiments start
mc_q = [conformal(sdm, f...; α=0.05) for f in kfold(sdm)]
q₊, q₋ = vec(median(vcat([hcat(q...) for q in mc_q]...), dims= 1))

Ŷ = predict(sdm, L; threshold=false)
uncertain = (x -> length(credibleclasses(x, q₊, q₋))).(Ŷ)
heatmap(predict(sdm, L), colormap=[:white, :darkgreen])
heatmap!(uncertain, colormap=[:transparent, :grey80])
contour!(predict(sdm, L), color=:darkgreen)
lines!(landmass, color=:black)
scatter!(presencelayer, color=:purple, markersize=4)
current_figure()
# normal code resumes

cs = cellarea(prd)

cmodel = deepcopy(sdm)

# Sensitivity analysis for the miscoverage rate
rlevels = LinRange(0.012, 0.2, 50)
miscoverage_holdout = holdout(cmodel)
qs = [conformal(cmodel, miscoverage_holdout...; α=u) for u in rlevels]

lines(rlevels, [q[1] for q in qs], label="Presence", color=:darkgreen)
lines!(rlevels, [q[2] for q in qs], label="Absence", color=:grey50, linestyle=:dash)
axislegend()
current_figure()

surf_presence = zeros(length(qs))
surf_undet = zeros(length(qs))
surf_unsure = zeros(length(qs))
surf_unsure_presence = zeros(length(qs))
surf_unsure_absence = zeros(length(qs))

𝐏 = predict(sdm; threshold=false)
eff = [mean(length.(credibleclasses.(𝐏, q...))) for q in qs]

scatter(rlevels, eff)

for i in eachindex(qs)
    Cp, Ca = credibleclasses(prd, qs[i]...)
    undet = .!(Cp .| Ca)
    sure_presence = Cp .& (.!Ca)
    unsure = Ca .& Cp
    unsure_presence = unsure .& distrib
    unsure_absence = unsure .& (.!distrib)
    surf_presence[i] = sum(mask(cs, nodata(sure_presence, false)))
    surf_undet[i] = sum(mask(cs, nodata(undet, false)))
    surf_unsure[i] = sum(mask(cs, nodata(unsure, false)))
    surf_unsure_presence[i] = sum(mask(cs, nodata(unsure_presence, false)))
    surf_unsure_absence[i] = sum(mask(cs, nodata(unsure_absence, false)))
end

# Cross-conformal with median range selected
mc_q = [conformal(sdm, f...; α=0.05) for f in kfold(sdm)]
q₊, q₋ = vec(median(vcat([hcat(q...) for q in mc_q]...), dims= 1))
Cp, Ca = credibleclasses(prd, q₊, q₋)

# Partition
sure_presence = Cp .& (.!Ca)
sure_absence = Ca .& (.!Cp)
unsure = Ca .& Cp
unsure_in = unsure .& distrib
unsure_out = unsure .& (.!distrib)

renderfigure("uncertainty")

# Example with unknown areas
mc_q = [conformal(sdm, f...; α=0.2) for f in kfold(sdm)]
q₊, q₋ = vec(median(vcat([hcat(q...) for q in mc_q]...), dims= 1))
Cp2, Ca2 = credibleclasses(prd, q₊, q₋)
undet = .!(Cp2 .| Ca2)

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
fprd = predict(sdm, F; threshold=false)
ft_distrib = predict(sdm, F; threshold=true)

mc_q = [conformal(sdm, f...; α=0.05) for f in kfold(sdm)]
q = vec(median(vcat([hcat(q...) for q in mc_q]...), dims= 1))
Cp, Ca = credibleclasses(prd, q₊, q₋)

fCp, fCa = credibleclasses(fprd, q...)

ft_sure_presence = fCp .& (.!fCa)
ft_sure_absence = fCa .& (.!fCp)
ft_unsure = fCa .& fCp
ft_unsure_in = ft_unsure .& ft_distrib
ft_unsure_out = ft_unsure .& (.!ft_distrib)

renderfigure("gainloss")

nv = novelty(L, F, variables(sdm))

renderfigure("novelty")