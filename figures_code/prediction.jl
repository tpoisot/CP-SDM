f = Figure(; size=(1200, 600))
ax1 = Axis(f[1,1]; aspect=DataAspect())
hm1 = heatmap!(ax1, prd, colormap=:linear_gow_60_85_c27_n256, colorrange=(0,1))
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
hm2 = heatmap!(ax2, bsvaria, colormap=:Greys, colorscale=log10)
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
        lines!(ax, p, color=:grey10)
    end
    contour!(ax, distrib, color=:red, levels=1)
end

Label(f[1, 1], "A", alignmode=Inside(), tellwidth=false, tellheight=false, valign=:top, halign=:left, fontsize=30)
Label(f[1, 2], "B", alignmode=Inside(), tellwidth=false, tellheight=false, valign=:top, halign=:left, fontsize=30)
current_figure()