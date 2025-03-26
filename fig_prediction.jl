
f = Figure(; size=(1200, 600))
ax1 = Axis(f[1,1]; aspect=DataAspect())
hm1 = heatmap!(ax1, prd, colormap=:navia, colorrange=(0,1))
Colorbar(
    f[1, 1],
    hm1;
    label = "Prediction (presence)",
    alignmode = Inside(),
    height = Relative(0.5),
    flipaxis = false,
    valign = :bottom,
    halign = :right,
    tellheight = false,
    tellwidth = false,
    vertical = true,
)
ax2 = Axis(f[1,2]; aspect=DataAspect())
hm2 = heatmap!(ax2, quantize(bsvaria, 100), colormap=:nuuk, colorrange=(0,1))
Colorbar(
    f[1, 2],
    hm2;
    label = "Inter-quantile range",
    alignmode = Inside(),
    height = Relative(0.5),
    flipaxis = false,
    valign = :bottom,
    halign = :right,
    tellheight = false,
    tellwidth = false,
    vertical = true,
)
for ax in [ax1, ax2]
    hidespines!(ax)
    hidedecorations!(ax)
    for p in polygons
        lines!(ax, p, color=:grey50)
    end
    contour!(ax, distrib, color=:red, levels=1)
end
CairoMakie.save(joinpath(fpath, "prediction.png"), current_figure())
current_figure()