
# Figure with projection and gain/Loss
f = Figure(; size=(1200, 600))
ax1 = Axis(f[1,1]; aspect=DataAspect())
ax2 = Axis(f[1,2]; aspect=DataAspect())
for ax in [ax1, ax2]
    for p in polygons
        poly!(ax, p, color=:grey90)
        lines!(ax, p, color=:grey10)
    end
end
heatmap!(ax1, nodata(ft_sure_presence, false), colormap=[:forestgreen])
heatmap!(ax1, nodata(ft_unsure, false), colormap=[:grey70])
cmap = [colorant"#fdb863", colorant"#e66101", colorant"#020202",colorant"#5e3c99", colorant"#b2abd2"]
heatmap!(ax2, nodata(sure_presence .& ft_sure_presence,  false), colormap=[cmap[3]])
heatmap!(ax2, nodata(sure_absence .& ft_unsure, false), colormap=[cmap[5]])
heatmap!(ax2, nodata(sure_presence .& ft_unsure, false), colormap=[cmap[1]])
heatmap!(ax2, nodata(unsure .& ft_unsure, false), colormap=[:grey70])
heatmap!(ax2, nodata((sure_absence .| unsure) .& ft_sure_presence, false), colormap=[cmap[4]]) # Certain gain
heatmap!(ax2, nodata((sure_presence .| unsure) .& ft_sure_absence, false), colormap=[cmap[2]]) # Certain loss
Legend(
    f[1, 2],
    alignmode = Inside(),
    height = Relative(0.4),
    valign = :bottom,
    halign = :right,
    tellheight = false,
    tellwidth = false,
    [PolyElement(; color = c) for c in [cmap..., :grey70]],
    ["Possible loss", "Sure loss", "Conserved", "Sure gain", "Possible gain", "Ambiguous"];
    orientation = :vertical,
    nbanks = 1,
    framevisible = false,
    vertical = false,
)
for ax in [ax1, ax2]
    for p in polygons
        lines!(ax, p, color=:grey10)
    end
    hidedecorations!(ax)
    hidespines!(ax)
end
contour!(ax1, ft_distrib, color=:red, levels=1)
Label(f[1, 1], "A", alignmode=Inside(), tellwidth=false, tellheight=false, valign=:top, halign=:left, fontsize=30)
Label(f[1, 2], "B ", alignmode=Inside(), tellwidth=false, tellheight=false, valign=:top, halign=:left, fontsize=30)
current_figure()