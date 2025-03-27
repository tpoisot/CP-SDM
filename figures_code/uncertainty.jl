
# Big figure with range and uncertainty

bins = LinRange(0.0, round(quantile(bsvaria, 0.9); digits=2), 80)
hiparams = (; bins=bins, normalization=:pdf)

f = Figure(; size=(1200, 600))
ax = Axis(f[1:2, 1]; aspect=DataAspect())
for p in polygons
    poly!(ax, p, color=:grey95)
    lines!(ax, p, color=:grey10)
end
heatmap!(ax, nodata(sure_presence, false), colormap=[:forestgreen])
heatmap!(ax, nodata(unsure, false), colormap=[:grey70])
hidespines!(ax)
hidedecorations!(ax)
for p in polygons
    lines!(ax, p, color=:grey10)
end
contour!(ax, distrib, color=:red, levels=1)
ax2 = Axis(f[1, 2], xlabel="Risk level α", ylabel="Range (km²)", yscale=log10)
scatter!(ax2, rlevels, clamp.(surf_presence .+ surf_unsure, 1, Inf), color=:grey70, label="Total range")
hlines!(ax2, [sum(mask(cs, nodata(distrib, false)))], color=:black, linestyle=:dash, label="SDM range")
scatter!(ax2, rlevels, clamp.(surf_presence, 1, Inf), color=:forestgreen, marker=:rect, label="Sure range")
axislegend(ax2, position=:rb)
ax3 = Axis(f[2, 2], xlabel="Inter-quantile range (ensemble)")
hist!(ax3, mask(bsvaria, nodata(sure_absence, false)); color=(:orange, 0.7), label="Sure absence", hiparams...)
hist!(ax3, mask(bsvaria, nodata(unsure, false)); color=(:grey80, 0.7), label="Unsure", hiparams...)
hist!(ax3, mask(bsvaria, nodata(sure_presence, false)); color=(:forestgreen, 0.7), label="Sure presence", hiparams...)
axislegend(ax3, nbanks=3)
tightlimits!(ax3)
hideydecorations!(ax3)
hidespines!(ax3, :r)
hidespines!(ax3, :l)
hidespines!(ax3, :t)
ylims!(ax2, 1e4, 1e6)
Label(f[1:2, 1], "A", alignmode=Inside(), tellwidth=false, tellheight=false, valign=:top, halign=:left, fontsize=30)
Label(f[1, 2], "B ", alignmode=Inside(), tellwidth=false, tellheight=false, valign=:top, halign=:right, fontsize=30)
Label(f[2, 2], "C ", alignmode=Inside(), tellwidth=false, tellheight=false, valign=:center, halign=:right, fontsize=30)
current_figure()