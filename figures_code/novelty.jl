
# Novelty map + histograms
f = Figure(; size=(1200, 600))
ax = Axis(f[1:3,1]; aspect=DataAspect())
hm = heatmap!(ax, nv, colormap=:linear_wcmr_100_45_c42_n256, colorscale=sqrt)
for p in polygons
    lines!(ax, p, color=:grey10, linewidth=1)
end
Colorbar(
    f[1:3, 1],
    hm;
    label = "Climatic novelty",
    alignmode = Inside(),
    height = Relative(0.4),
    flipaxis = false,
    valign = :bottom,
    halign = :right,
    tellheight = false,
    tellwidth = false,
    vertical = true,
)
hidespines!(ax)
hidedecorations!(ax)
ax_sa = Axis(f[1,2]; ylabel="Sure absence")
ax_us = Axis(f[2,2]; ylabel="Unsure")
ax_sp = Axis(f[3,2]; ylabel="Sure presence")
hist!(ax_sa, mask(nv, nodata(ft_sure_absence, false)), bins=LinRange(0.1, 1.5, 80), color=(:orange, 0.7))
hist!(ax_us, mask(nv, nodata(ft_unsure, false)), bins=LinRange(0.1, 1.5, 80), color=(:grey60, 0.7))
hist!(ax_sp, mask(nv, nodata(ft_sure_presence, false)), bins=LinRange(0.1, 1.5, 80), color=(:forestgreen, 0.7))
for ax in [ax_sa, ax_sp, ax_us]
    xlims!(ax, (0.1, 1.0))
    hideydecorations!(ax, label=false)
    tightlimits!(ax)
end

Label(f[1:3, 1], "A", alignmode=Inside(), tellwidth=false, tellheight=false, valign=:top, halign=:left, fontsize=30)
Label(f[1, 2], "B ", alignmode=Inside(), tellwidth=false, tellheight=false, valign=:top, halign=:right, fontsize=30)
Label(f[2, 2], "C ", alignmode=Inside(), tellwidth=false, tellheight=false, valign=:top, halign=:right, fontsize=30)
Label(f[3, 2], "D ", alignmode=Inside(), tellwidth=false, tellheight=false, valign=:top, halign=:right, fontsize=30)

current_figure()
