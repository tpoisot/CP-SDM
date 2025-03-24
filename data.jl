if ~isfile("occurrences.csv")
    Downloads.download("https://raw.githubusercontent.com/tpoisot/ConformalSDM/refs/heads/main/data/occurrences.csv", "occurrences.csv")
end

occlines = readlines("occurrences.csv")[2:end]
class_a = filter(contains("Class A"), occlines)
valid_info = [split(obs, ",")[end-2:end] for obs in class_a]

datefun = (s) -> Dates.Date(s, Dates.dateformat"yyyy-mm-ddTH:M:SZ")

records = [Occurrence(what="Bigfoot", when=datefun(r[1]), where=(parse(Float64, r[3]), parse(Float64, r[2]))) for r in valid_info]

polygons = [
    SpeciesDistributionToolkit.gadm("USA", "Oregon"),
    #SpeciesDistributionToolkit.openstreetmap("California"),
    #SpeciesDistributionToolkit.openstreetmap("Nevada"),
    SpeciesDistributionToolkit.gadm("USA", "Washington")
]

extent = SpeciesDistributionToolkit._reconcile(SpeciesDistributionToolkit.boundingbox.(polygons))

provider = RasterData(WorldClim2, BioClim)
prj = Projection(SSP370, MRI_ESM2_0)
L = SDMLayer{Float32}[SDMLayer(provider; resolution=2.5, layer=l, extent...) for l in layers(provider)]
F = SDMLayer{Float32}[SDMLayer(provider, prj; resolution=2.5, layer=l, extent...) for l in layers(provider)]

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
