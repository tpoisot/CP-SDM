# Generate a bivariate palette
function _palette(; low=colorant"#e8e8e8", high=colorant"#120fe3", breaks=3)
    breakpoints = LinRange(0.0, 1.0, breaks)
    return ColorSchemes.weighted_color_mean.(breakpoints, high, low)
end

# Record the layers by quantiles
function discretize(layer, n::Integer)
    return (x -> round(Int64, x)).(rescale(layer, 1, n))
end

# Get the palettes
nbreaks = 5

# 
low = colorant"#e8e8e8"
h1 = colorant"#be64ac"
h2 = colorant"#5ac8c8"

colormap1 = _palette(; low=low, high=h1, breaks=nbreaks)
colormap2 = _palette(; low=low, high=h2, breaks=nbreaks)
colormatrix = [ColorBlendModes.blend.(c1, c2; mode=BlendMultiply) for c1 in colormap1, c2 in colormap2]

# Discrete maps
m1 = discretize(current_shift, nbreaks)
m2 = discretize(future_shift, nbreaks)
category = similar(m1)
for i in eachindex(category)
    category[i] = LinearIndices(colormatrix)[m1[i], m2[i]]
end

f = Figure(; size=(1200, 600))
ax = Axis(f[1:2, 1]; aspect=DataAspect())
heatmap!(ax, category, colormap=vec(colormatrix))
lines!(ax, landmass, color=:black)

ax_leg = Axis(f[2, 2]; aspect=DataAspect())
xlims!(ax_leg, -2.2, 2.2)

xp = LinRange(-1, 1, size(colormatrix, 1) + 1)
yp = LinRange(-1, 1, size(colormatrix, 2) + 1)

θ = -π / 4
for i in axes(colormatrix, 1)
    xc = (xp[i], xp[i+1]) .+ (0.015, -0.015)
    for j in axes(colormatrix, 2)
        yc = (yp[j], yp[j+1]) .+ (0.015, -0.015)
        corners = [(xc[1], yc[1]), (xc[2], yc[1]), (xc[2], yc[2]), (xc[1], yc[2])]
        r_corners = [
            (c[1] * cos(θ) - c[2] * sin(θ), c[2] * cos(θ) + c[1] * sin(θ))
            for c in corners
        ]
        poly!(ax_leg, r_corners, color=colormatrix[i, j], strokecolor=:black, strokewidth=0.5)
    end
end

function makelab!(ax, start, end1, end2, label; path=Ann.Paths.Corner(), style=Ann.Styles.LineArrow(), labelspace=:data, kwargs...)
    sx = (start[1] * cos(θ) - start[2] * sin(θ), start[2] * cos(θ) + start[1] * sin(θ))
    e1x = (end1[1] * cos(θ) - end1[2] * sin(θ), end1[2] * cos(θ) + end1[1] * sin(θ))
    e2x = (end2[1] * cos(θ) - end2[2] * sin(θ), end2[2] * cos(θ) + end2[1] * sin(θ))
    annotation!(ax, sx..., e1x...;
        text=label,
        path=path,
        style=style,
        labelspace=labelspace,
        kwargs...
    )
    annotation!(ax, sx..., e2x...;
        text=label,
        path=path,
        style=style,
        labelspace=labelspace,
        kwargs...
    )
end

makelab!(ax_leg, (0, 1.5), (1, 1), (-1, 1), "Future conditions\nless represented")
makelab!(ax_leg, (1.5, 0), (1, 1), (1, -1), "Current conditions\nless represented")
makelab!(ax_leg, (-1.5, 0), (-1, -1), (-1, 1), "Current conditions\nwell represented")
makelab!(ax_leg, (0, -1.5), (-1, -1), (1, -1), "Future conditions\nwell represented")

f

xlabs = ([1, 2, 3, 4], ["Uncertain\n\t→ Absent", "Uncertain\n\t→ Uncertain", "Present\n\t→ Uncertain", "Present\n\t→ Present"])
ax_unc = Axis(f[1, 2], xticks = xlabs)

categories = Int64[]
values = Float64[]
dodge = Int64[]

u_u = nodata((current_uncertainty .== 1) .* (future_uncertainty .== 1), false)
u_a = nodata((current_uncertainty .== 1) .* (future_uncertainty .== 0), false)
p_u = nodata((current_uncertainty .== 2) .* (future_uncertainty .== 1), false)
p_p = nodata((current_uncertainty .== 2) .* (future_uncertainty .== 2), false)

for (i, l) in enumerate([u_a, u_u, p_u, p_p])
    future_shift.x = l.x
    future_shift.y = l.y
    current_shift.x = l.x
    current_shift.y = l.y
    append!(categories, fill(i, 2length(l)))
    append!(values, mask(future_shift, l))
    append!(values, mask(current_shift, l))
    append!(dodge, fill(1, length(l)))
    append!(dodge, fill(2, length(l)))
end



boxplot!(ax_unc,
    categories,
    values,
    dodge=dodge,
    show_notch=false,
    color=:white,#map(d -> d == 1 ? (h1, 0.1) : (h2, 0.1), dodge),
    strokecolor=[h1, h2, h1, h2, h1, h2, h1, h2],
    mediancolor=[h1, h2, h1, h2, h1, h2, h1, h2],
    strokewidth=2,
    medianlinewidth=4, 
    gap=0.4,
    show_outliers=false,
)

hidespines!(ax_leg)
hidedecorations!(ax_leg)
hidespines!(ax)
hidedecorations!(ax)

hidespines!(ax_unc)

f