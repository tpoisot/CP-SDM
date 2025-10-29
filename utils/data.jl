using SpeciesDistributionToolkit
const SDT = SpeciesDistributionToolkit
using CairoMakie
import Random
import Dates
Random.seed!(42069)

# Load the data from the occurrences interface package - this is the entire
# dataset, but we don't particularly need to clip it at this point
records = OccurrencesInterface.__demodata();

# Get the country-level data from GADM
gadm_usa_level1 = getpolygon(PolygonData(GADM, Countries); level=1, country="USA")

# Now we get the states we want to include in the model
polygons = [
    gadm_usa_level1["California"],
    gadm_usa_level1["Idaho"],
    gadm_usa_level1["Nevada"],
    gadm_usa_level1["Oregon"],
    gadm_usa_level1["Washington"],
]

# We merge the states (but keep the borders for the map)
landmass = FeatureCollection(polygons)

# Bounding box to clip the layers, with a padding to generate the
# pseudo-absences - we will use the strict limit of the polygons later on
extent = SDT.boundingbox(landmass; padding=2.0)

# Get the layers
provider = RasterData(WorldClim2, BioClim)

# We start by getting a template layer to generate the pseudo-absences and the presence layer
M = SDMLayer(provider; resolution=2.5, layer=1, extent...)
presencelayer = mask(M, records)
distance = pseudoabsencemask(DistanceToEvent, presencelayer)
presencelayer = trim(mask!(presencelayer, landmass))
distance = trim(mask!(distance, landmass))
background = nodata(distance, d -> !(20 <= d <= 250))
absencelayer = backgroundpoints(background, 3sum(presencelayer))

# Test
lines(landmass)
scatter!(presencelayer)
scatter!(absencelayer)
current_figure()

# Now we get the full bioclim variables for the actual study area
extent = SDT.boundingbox(landmass)
L = SDMLayer{Float32}[SDMLayer(provider; resolution=2.5, layer=l, extent...) for l in layers(provider)]
L = trim.(mask!(L, landmass))

# At this point, we can run the SDM

# Set up the model - logistic regression with Z-score before
sdm = SDM(ZScore, Logistic, L, presencelayer, absencelayer)
hyperparameters!(classifier(sdm), :η, 1e-3) # Slow descent
hyperparameters!(classifier(sdm), :interactions, :all) # All interactions
hyperparameters!(classifier(sdm), :epochs, 10000) # Longer training

# Train the model with optimal set of variables, using forward selection and MCC
# as the measure - this will return a trained model
folds = kfold(sdm)
variables!(sdm, ForwardSelection, folds; verbose=true)
threshold!(sdm)

# We save the model - we will re-use it later on
SDeMo.writesdm("artifacts/sdm.json", sdm)

# And we also save the historical climate data
SimpleSDMLayers.save("artifacts/historical.tif", L)

# Future layers
GCMs = [MRI_ESM2_0, ACCESS_CM2, EC_Earth3_Veg, CanESM5, GFDL_ESM4, MIROC6]

for gcm in GCMs
    prj = Projection(SSP370, gcm)
    tf = SDMLayer{Float32}[SDMLayer(provider, prj; resolution=2.5, timespan=Dates.Year(2081) => Dates.Year(2100), layer=l, extent...) for l in layers(provider)]
    
    # Important !!!!
    tf = trim.(mask!(tf, landmass))

    # Mask the future layers
    for i in eachindex(tf)
        tf[i].indices .&= L[1].indices
        tf[i].x = L[1].x
        tf[i].y = L[1].y
    end

    SimpleSDMLayers.save("artifacts/future-SSP370-$(gcm).tif", tf)
end



# Code for novelty - we take the median value for each pixel here purely to save
# time, otherwise this is a very long step for very minor differences in the end


include(joinpath(pwd(), "utils/novelty.jl"))
future_climates = filter(contains("future"), readdir("artifacts/"; join=true))

F = [
    [SimpleSDMLayers._read_geotiff(future_file; bandnumber=i) for i in 1:19] for future_file in future_climates
]


Fmed = [mosaic(median, [F[m][i] for m in keys(F)]) for i in eachindex(L)]

novel_climates = novelty(Fmed, sdm)
lost_climates = novelty(L, sdm)
SimpleSDMLayers.save("artifacts/novelty.tif", [novel_climates, lost_climates])