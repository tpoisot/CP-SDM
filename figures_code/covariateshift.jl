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
m1 = discretize(novel_climates, nbreaks)
m2 = discretize(lost_climates, nbreaks)
category = similar(m1)
for i in eachindex(category)
    category[i] = LinearIndices(colormatrix)[m1[i], m2[i]]
end

f = Figure(; size=(1200, 600))
ax = Axis(f[1:5, 1:2]; aspect=DataAspect())
heatmap!(ax, category, colormap=vec(colormatrix))
lines!(ax, landmass, color=:black)

ax_leg = Axis(f[4:5, 3:4]; aspect=DataAspect())

xp = LinRange(-1, 1, size(colormatrix, 1)+1)
yp = LinRange(-1, 1, size(colormatrix, 2)+1)

θ = π / 4
for i in axes(colormatrix, 1)
    xc = (xp[i], xp[i+1]) .+ (0.015, -0.015)
    for j in axes(colormatrix, 2)
        yc = (yp[j], yp[j+1]) .+ (0.015, -0.015)
        corners = [(xc[1], yc[1]), (xc[2], yc[1]), (xc[2], yc[2]), (xc[1], yc[2])]
        r_corners = [
            (c[1] * cos(θ) - c[2] * sin(θ), c[2] * cos(θ) + c[1] * sin(θ))
            for c in corners
        ]
        #scatter!(ax_leg, [xr], [yr], color=colormatrix[i,j])
        poly!(ax_leg, r_corners, color=colormatrix[i,j], strokecolor=:black, strokewidth=0.5)
    end
end

annotation!(ax_leg, 1.1, 1.1, 0, sqrt(2),
    text = "Novel climates\nemerge",
    path = Ann.Paths.Corner(),
    style = Ann.Styles.LineArrow(),
    labelspace = :data
)

annotation!(ax_leg, 1.1, 1.1, sqrt(2), 0,
    text = "Novel climates\nemerge",
    path = Ann.Paths.Corner(),
    style = Ann.Styles.LineArrow(),
    labelspace = :data
)

annotation!(ax_leg, 1.1, -1.1, sqrt(2), 0,
    text = "Hist. climates\nremain",
    path = Ann.Paths.Corner(),
    style = Ann.Styles.LineArrow(),
    labelspace = :data
)

annotation!(ax_leg, 1.1, -1.1, 0, -sqrt(2),
    text = "Hist. climates\nremain",
    path = Ann.Paths.Corner(),
    style = Ann.Styles.LineArrow(),
    labelspace = :data
)

annotation!(ax_leg, -1.1, -1.1, 0, -sqrt(2),
    text = "No novel\nclimates",
    path = Ann.Paths.Corner(),
    style = Ann.Styles.LineArrow(),
    labelspace = :data
)

annotation!(ax_leg, -1.1, -1.1, -sqrt(2), 0,
    text = "No novel\nclimates",
    path = Ann.Paths.Corner(),
    style = Ann.Styles.LineArrow(),
    labelspace = :data
)

annotation!(ax_leg, -1.1, 1.1, 0, sqrt(2),
    text = "Hist. climates\nare lost",
    path = Ann.Paths.Corner(),
    style = Ann.Styles.LineArrow(),
    labelspace = :data
)

annotation!(ax_leg, -1.1, 1.1, -sqrt(2), 0,
    text = "Hist. climates\nare lost",
    path = Ann.Paths.Corner(),
    style = Ann.Styles.LineArrow(),
    labelspace = :data
)

f

ax_unc = Axis(f[1:3, 3:4])

categories = Int64[]
values = Float64[]
dodge = Int64[]



u_u = nodata((current_uncertainty.==1) .* (future_uncertainty.==1), false)
u_a = nodata((current_uncertainty.==1) .* (future_uncertainty.==0), false)
p_u = nodata((current_uncertainty.==2) .* (future_uncertainty.==1), false)
p_p = nodata((current_uncertainty.==2) .* (future_uncertainty.==2), false)

for (i, l) in enumerate([u_a, u_u, p_u, p_p])
    append!(categories, fill(i, 2length(l)))
    append!(values, mask(lost_climates, l))
    append!(values, mask(novel_climates, l))
    append!(dodge, fill(1, length(l)))
    append!(dodge, fill(2, length(l)))
end

boxplot!(ax_unc, categories, values, dodge = dodge, show_notch = false, color = map(d->d==1 ? h2 : h1, dodge))

hidespines!(ax_leg)
hidedecorations!(ax_leg)
hidespines!(ax)
hidedecorations!(ax)

f