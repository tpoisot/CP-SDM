if ~isfile("occurrences.csv")
    Downloads.download("https://raw.githubusercontent.com/tpoisot/ConformalSDM/refs/heads/main/data/occurrences.csv", "occurrences.csv")
end

occlines = readlines("occurrences.csv")[2:end]
class_a = filter(contains("Class A"), occlines)
valid_info = [split(obs, ",")[end-2:end] for obs in class_a]

datefun = (s) -> Dates.Date(s, Dates.dateformat"yyyy-mm-ddTH:M:SZ")

records = [Occurrence(what="Bigfoot", when=datefun(r[1]), where=(parse(Float64, r[3]), parse(Float64, r[2]))) for r in valid_info]

polygons = [
    SpeciesDistributionToolkit.openstreetmap("Oregon"),
    SpeciesDistributionToolkit.openstreetmap("California"),
    SpeciesDistributionToolkit.openstreetmap("Nevada"),
    SpeciesDistributionToolkit.openstreetmap("Washington state")
]

extent = SpeciesDistributionToolkit._reconcile(SpeciesDistributionToolkit.boundingbox.(polygons))

provider = RasterData(WorldClim2, BioClim)
L = SDMLayer{Float32}[SDMLayer(provider; resolution=5.0, layer=l, extent...) for l in layers(provider)]

lmask = [mask(L[1], p) for p in polygons]
msk = reduce(.|, [lm.indices for lm in lmask])

for i in eachindex(L)
    L[i].indices = msk
end