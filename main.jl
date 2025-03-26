using CairoMakie
using SpeciesDistributionToolkit
using Statistics
import Downloads
import Dates
import PrettyTables
import Random
Random.seed!(42069)

# Load the functions we need here
include("utils/theme.jl")
include("utils/lib.jl")
include("utils/cellsize.jl")
include("utils/novelty.jl")
include("utils/data.jl")

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
δ = 1.05 # Tapering for distance
presencelayer = mask(first(L), Occurrences(records))
background = pseudoabsencemask(DistanceToEvent, presencelayer)
bgpoints = backgroundpoints(nodata(background, d -> d < 10).^δ, 3sum(presencelayer))

# Set up the model - logistic regression with Z-score before
sdm = SDM(ZScore, Logistic, L, presencelayer, bgpoints)
hyperparameters!(classifier(sdm), :η, 1e-3) # Slow descent
hyperparameters!(classifier(sdm), :interactions, :all) # All interactions
hyperparameters!(classifier(sdm), :epochs, 8000) # Longer training

# Folds
folds = kfold(sdm)

# Train the model with optimal set of variables, using forward selection and MCC
# as the measure
variables!(sdm, ForwardSelection, folds; verbose=true)


# VI
vi = variableimportance(sdm, kfold(sdm); threshold=false)
miv = variables(sdm)[last(findmax(vi))]

renderfigure("occurrences")

# Measure of model performance
# Make a PrettyTable for output
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

renderfigure("prediction")

cs = cellsize(prd)

cmodel = deepcopy(sdm)

# Sensitivity analysis for the miscoverage rate
rlevels = LinRange(0.01, 0.2, 250)
qs = [_estimate_q(cmodel, holdout(cmodel)...; α=u) for u in rlevels]
surf_presence = zeros(length(qs))
surf_undet = zeros(length(qs))
surf_unsure = zeros(length(qs))
surf_unsure_presence = zeros(length(qs))
surf_unsure_absence = zeros(length(qs))

𝐏 = predict(sdm; threshold=false)
eff = [mean(length.(credibleclasses.(𝐏, q))) for q in qs]

for i in eachindex(qs)
    Cp, Ca = credibleclasses(prd, qs[i])
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
q = median([_estimate_q(cmodel, fold...; α=0.05) for fold in kfold(cmodel; k=10)])
Cp, Ca = credibleclasses(prd, q)

# Partition
sure_presence = Cp .& (.!Ca)
sure_absence = Ca .& (.!Cp)
unsure = Ca .& Cp
unsure_in = unsure .& distrib
unsure_out = unsure .& (.!distrib)

renderfigure("uncertainty")

# Example with unknown areas
q2 = median([_estimate_q(cmodel, fold...; α=0.2) for fold in kfold(cmodel; k=10)])
Cp2, Ca2 = credibleclasses(prd, q2)
undet = .!(Cp2 .| Ca2)

renderfigure("undetrange")

# Shapley values
S = explain(sdm, L; threshold=false)

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

fCp, fCa = credibleclasses(fprd, q)

ft_sure_presence = fCp .& (.!fCa)
ft_sure_absence = fCa .& (.!fCp)
ft_unsure = fCa .& fCp
ft_unsure_in = ft_unsure .& ft_distrib
ft_unsure_out = ft_unsure .& (.!ft_distrib)

renderfigure("gainloss")

nv = novelty(L, F, variables(sdm))

renderfigure("novelty")