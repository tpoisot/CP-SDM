# Figure on range undeterminacy
f = Figure(; size=(1200, 600))
ax = Axis(f[1:2, 2]; aspect=DataAspect())
for p in polygons
    poly!(ax, p, color=:grey95)
    lines!(ax, p, color=:grey10)
end
heatmap!(ax, nodata(undet, false), colormap=[:grey50])
for p in polygons
    lines!(ax, p, color=:grey10)
end
contour!(ax, distrib, color=:red, levels=1)
hidespines!(ax)
hidedecorations!(ax)
ax2 = Axis(f[1, 1], xlabel="Risk level α", ylabel="Inefficiency")
hlines!(ax2, [1.0], color=:grey50, linestyle=:dash)
scatter!(ax2, rlevels, eff, color=:black)
ax3 = Axis(f[2, 1], xlabel="Risk level α", ylabel="Undetermined range (km²)", yscale=log10)
surf_undet[findall(iszero, surf_undet)] .= NaN
scatter!(ax3, rlevels, surf_undet, color=:black)
xlims!(ax3, extrema(rlevels))
Label(f[1, 1], " A ", alignmode=Inside(), tellwidth=false, tellheight=false, valign=:bottom, halign=:left, fontsize=30)
Label(f[2, 1], " B ", alignmode=Inside(), tellwidth=false, tellheight=false, valign=:top, halign=:left, fontsize=30)
Label(f[1:2, 2], "C ", alignmode=Inside(), tellwidth=false, tellheight=false, valign=:top, halign=:right, fontsize=30)
current_figure()