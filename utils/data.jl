# Load the data from the occurrences interface
records = OccurrencesInterface.__demodata();

# Get the country-level data from GADM
gadm_usa_level1 = getpolygon(PolygonData(GADM, Countries); level=1, country="USA")

# Now we get the states we want to include in the model
polygons = [
    gadm_usa_level1["California"],
    #gadm_usa_level1["Idaho"],
    #gadm_usa_level1["Nevada"],
    gadm_usa_level1["Oregon"],
    gadm_usa_level1["Washington"],
]

# We merge the states (but keep the borders for the map)
landmass = FeatureCollection(polygons)

# Bounding box to clip the layers
extent = SDT.boundingbox(landmass)

# Get the layers
provider = RasterData(WorldClim2, BioClim)
prj = Projection(SSP370, MRI_ESM2_0)
L = SDMLayer{Float32}[SDMLayer(provider; resolution=2.5, layer=l, extent...) for l in layers(provider)]
F = SDMLayer{Float32}[SDMLayer(provider, prj; resolution=2.5, timespan=Dates.Year(2081) => Dates.Year(2100), layer=l, extent...) for l in layers(provider)]

# Mask the layers
mask!(L, landmass)

# Mask the future layers
for i in eachindex(F)
    F[i].indices .&= L[1].indices
    F[i].x = L[1].x
    F[i].y = L[1].y
end
