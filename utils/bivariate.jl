
# Generate a bivariate palette
function _palette(; low=colorant"#e8e8e8", high=colorant"#120fe3", breaks=3)
    breakpoints = LinRange(0.0, 1.0, breaks)
    return ColorSchemes.weighted_color_mean.(breakpoints, high, low)
end

# Get the palettes
nbreaks = 100

# 
low = colorant"#faf0d1"
h1 = colorant"#4080bf"
h2 = colorant"#40bf40"

colormap1 = _palette(; low=low, high=h1, breaks=nbreaks)
colormap2 = _palette(; low=low, high=h2, breaks=nbreaks)
colormatrix = [ColorBlendModes.blend.(c1, c2; mode=BlendDarken) for c1 in colormap1, c2 in colormap2]

# Record the layers by quantiles
function discretize(layer, n::Integer)
    return (x -> round(Int64, x)).(rescale(layer, 1, n))
end

# Discrete maps
m1 = discretize(quantize(P), nbreaks)
m2 = discretize(quantize(U), nbreaks)
category = similar(m1)
for i in eachindex(category)
    category[i] = LinearIndices(colormatrix)[m1[i], m2[i]]
end

f = Figure()
ax = [Axis(f[x...]; aspect=DataAspect()) for x in [(1, 1), (1, 2), (2, 1)]]
heatmap!(ax[1], quantize(P), colormap=colormap1)
heatmap!(ax[2], quantize(U), colormap=colormap2)
heatmap!(ax[3], category, colormap=vec(colormatrix))
f

