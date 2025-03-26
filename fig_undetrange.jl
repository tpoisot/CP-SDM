# Figure on range undeterminacy
f = Figure(; size=(1200, 600))
ax = Axis(f[1:2,1]; aspect=DataAspect())
for p in polygons
    poly!(ax, p, color=:grey95)
    lines!(ax, p, color=:grey10)
end
heatmap!(ax, nodata(undet, false), colormap=[:transparent, :red])
contour!(ax, distrib, color=:red, levels=1)
hidespines!(ax)
hidedecorations!(ax)
ax2 = Axis(f[1,2], xlabel="Risk level α", ylabel="Efficiency")
hlines!(ax2, [1.0], color=:grey50, linestyle=:dash)
scatter!(ax2, rlevels, eff, color=:black)
CairoMakie.save(joinpath(fpath, "undetrange.png"), current_figure())
current_figure()