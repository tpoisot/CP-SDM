
# Plot the figure for presence/absence
f = Figure(; size=(600, 600))
ax = Axis(f[1,1]; aspect=DataAspect())
for p in polygons
    poly!(ax, p, color=:grey95)
    lines!(ax, p, color=:grey10)
end
scatter!(ax, presencelayer, color=:white, strokecolor=:forestgreen, strokewidth=2)
scatter!(ax, bgpoints, color=:grey30, markersize=4)
hidespines!(ax)
hidedecorations!(ax)
CairoMakie.save(joinpath(fpath, "occurrrences.png"), current_figure())
current_figure()