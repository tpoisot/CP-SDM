#=
if ~isfile("occurrences.csv")
    Downloads.download("https://raw.githubusercontent.com/tpoisot/ConformalSDM/refs/heads/main/data/occurrences.csv", "occurrences.csv")
end

occlines = readlines("occurrences.csv")[2:end]
class_a = filter(contains("Class A"), occlines)
valid_info = [split(obs, ",")[end-2:end] for obs in class_a]

datefun = (s) -> Dates.Date(s, Dates.dateformat"yyyy-mm-ddTH:M:SZ")
=#

records = OccurrencesInterface.__demodata();

gadm_usa_level1 = getpolygon(PolygonData(GADM, Countries); level=1, country="USA")

# Get the states
polygons = [
    gadm_usa_level1["Oregon"],
    gadm_usa_level1["California"],
    gadm_usa_level1["Nevada"],
    gadm_usa_level1["Idaho"],
    gadm_usa_level1["Washington"]
]

# We merge the states (but keep the borders)
landmass = vcat(polygons...)

extent = SDT.boundingbox(landmass)

provider = RasterData(WorldClim2, BioClim)
prj = Projection(SSP370, MRI_ESM2_0)
L = SDMLayer{Float32}[SDMLayer(provider; resolution=2.5, layer=l, extent...) for l in layers(provider)]
F = SDMLayer{Float32}[SDMLayer(provider, prj; resolution=2.5, timespan=Dates.Year(2081) => Dates.Year(2100), layer=l, extent...) for l in layers(provider)]

lmask = [mask(L[1], p) for p in polygons]
msk = reduce(.|, [lm.indices for lm in lmask])

for i in eachindex(L)
    L[i].indices = msk
end

for i in eachindex(F)
    F[i].indices = msk
    F[i].x = L[1].x
    F[i].y = L[1].y
end
