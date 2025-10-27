# Generate a bivariate palette
function _palette(; low=colorant"#e8e8e8", high=colorant"#120fe3", breaks=3)
    breakpoints = LinRange(0.0, 1.0, breaks)
    return ColorSchemes.weighted_color_mean.(breakpoints, high, low)
end

# Get the palettes
nbreaks = 3

# 
low = colorant"#ffffff"
h1 = colorant"#12A500"
h2 = colorant"#9200A5"

colormap1 = _palette(; low=low, high=h1, breaks=nbreaks)
colormap2 = _palette(; low=low, high=h2, breaks=nbreaks)
colormatrix = [ColorBlendModes.blend.(c1, c2; mode=BlendDarken) for c1 in colormap1, c2 in colormap2]

# Discrete maps
m1 = current_uncertainty .+ 1
m2 = future_uncertainty .+ 1
category = similar(m1)
for i in eachindex(category)
    category[i] = LinearIndices(colormatrix)[m1[i], m2[i]]
end

f = Figure(; size=(1200, 600))
ax1 = Axis(f[1, 1]; aspect=DataAspect())
ax2 = Axis(f[1, 2]; aspect=DataAspect())
heatmap!(ax1, category, colormap=vec(colormatrix))
lines!(ax1, landmass, color=:black)
f

total_area = sum(cell_surface)
current_bars = [sum(mask(cell_surface, nodata(current_uncertainty .== i, false))) for i in 0:2]
future_bars = [sum(mask(cell_surface, nodata(future_uncertainty .== i, false))) for i in 0:2]
current_bars ./= total_area
future_bars ./= total_area

# Get the transition matrix
trs = zeros(Float64, 3, 3)
for cur in [0, 1, 2]
    this_cur = current_uncertainty .== cur
    for fut in [0, 1, 2]
        this_fut = future_uncertainty .== fut
        this_tr_area = sum(mask(cell_surface, nodata(this_cur .& this_fut, false)))
        trs[cur+1, fut+1] = this_tr_area
    end
end
trs ./= total_area

cur_bars = vcat(0.0, current_bars .* total_area...)
ccur_bars = cumsum(cur_bars)
fut_bars = vcat(0.0, future_bars .* total_area...)
cfut_bars = cumsum(fut_bars)

for i in 1:3
    for j in 1:3

        tr = trs[i, j] * total_area

        left_bottom = ccur_bars[i] / total_area
        right_bottom = cfut_bars[j] / total_area
        left_top = ccur_bars[i] / total_area + cur_bars[i+1] / total_area * (tr / cur_bars[i+1])
        right_top = cfut_bars[j] / total_area + fut_bars[j+1] / total_area * (tr / fut_bars[j+1])

        # Transition bottom

        quad_easing(x) = x < 0.5 ? 2 * (x^2) : 1 - (-2 * x + 2)^2 / 2
        vshift = quad_easing.(LinRange(0.0, 1.0, 10))
        Δb = right_bottom - left_bottom
        Δt = right_top - left_top
        bottomshift = left_bottom .+ Δb .* vshift
        topshift = left_top .+ Δt .* vshift
        band!(ax2, LinRange(0.1, 0.9, length(vshift)), bottomshift, topshift, color=colormatrix[i, j], alpha=0.5)
    end
end

poly!(ax2, Point2f[(0, 0), (0.1, 0), (0.1, current_bars[1]), (0, current_bars[1])], color=colormap1[1], strokecolor=:black, strokewidth=1)
poly!(ax2, Point2f[(0, current_bars[1]), (0.1, current_bars[1]), (0.1, current_bars[1] + current_bars[2]), (0, current_bars[1] + current_bars[2])], color=colormap1[2], strokecolor=:black, strokewidth=1)
poly!(ax2, Point2f[(0, current_bars[1] + current_bars[2]), (0.1, current_bars[1] + current_bars[2]), (0.1, 1), (0, 1)], color=colormap1[3], strokecolor=:black, strokewidth=1)

poly!(ax2, Point2f[(1, 0), (0.9, 0), (0.9, future_bars[1]), (1, future_bars[1])], color=colormap2[1], strokecolor=:black, strokewidth=1)
poly!(ax2, Point2f[(1, future_bars[1]), (0.9, future_bars[1]), (0.9, future_bars[1] + future_bars[2]), (1, future_bars[1] + future_bars[2])], color=colormap2[2], strokecolor=:black, strokewidth=1)
poly!(ax2, Point2f[(1, future_bars[1] + future_bars[2]), (0.9, future_bars[1] + future_bars[2]), (0.9, 1), (1, 1)], color=colormap2[3], strokecolor=:black, strokewidth=1)
f

