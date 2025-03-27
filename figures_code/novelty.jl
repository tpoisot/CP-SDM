tr_conserved = nodata(sure_presence .& ft_sure_presence, false)
tr_maybegain =  nodata(sure_absence .& ft_unsure, false)
tr_maybeloss = nodata(sure_presence .& ft_unsure, false)
tr_ambig = nodata(unsure .& ft_unsure, false)
tr_suregain = nodata((sure_absence .| unsure) .& ft_sure_presence, false)
tr_sureloss = nodata((sure_presence .| unsure) .& ft_sure_absence, false)

# Novelty map + histograms
f = Figure(; size=(1200, 600))
ax = Axis(f[1:3, 1:2]; aspect=DataAspect())
hm = heatmap!(ax, nv, colormap=:linear_wcmr_100_45_c42_n256, colorscale=sqrt)
for p in polygons
    lines!(ax, p, color=:grey10, linewidth=1)
end
Colorbar(
    f[1:3, 1:2],
    hm;
    label="Climatic novelty",
    alignmode=Inside(),
    height=Relative(0.4),
    flipaxis=false,
    valign=:bottom,
    halign=:right,
    tellheight=false,
    tellwidth=false,
    vertical=true,
)
hidespines!(ax)
hidedecorations!(ax)
ax_cr = Axis(f[1, 3]; ylabel="Conserved", xscale=sqrt)
ax_am = Axis(f[1, 4]; ylabel="Ambiguous", xscale=sqrt)
ax_mg = Axis(f[2, 3]; ylabel="Possible gain", xscale=sqrt)
ax_ml = Axis(f[2, 4]; ylabel="Possible loss", xscale=sqrt)
ax_sg = Axis(f[3, 3]; ylabel="Sure gain", xscale=sqrt)
ax_sl = Axis(f[3, 4]; ylabel="Sure loss", xscale=sqrt)
hist!(ax_cr, mask(nv, tr_conserved), bins=LinRange(0.1, 1.5, 80), color=(:black, 1.0))
hist!(ax_am, mask(nv, tr_ambig), bins=LinRange(0.1, 1.5, 80), color=(:grey60, 1.0))
hist!(ax_mg, mask(nv, tr_maybegain), bins=LinRange(0.1, 1.5, 10), color=(cmap[5], 1.0))
hist!(ax_sg, mask(nv, tr_suregain), bins=LinRange(0.1, 1.5, 80), color=(cmap[4], 1.0))
hist!(ax_ml, mask(nv, tr_maybeloss), bins=LinRange(0.1, 1.5, 80), color=(cmap[1], 1.0))
hist!(ax_sl, mask(nv, tr_sureloss), bins=LinRange(0.1, 1.5, 80), color=(cmap[2], 1.0))
for ax in [ax_cr, ax_am, ax_mg, ax_ml, ax_sg, ax_sl]
    xlims!(ax, (0.1, 1.5))
    hideydecorations!(ax, label=false)
    tightlimits!(ax)
end

Label(f[1:3, 1:2], "A", alignmode=Inside(), tellwidth=false, tellheight=false, valign=:top, halign=:left, fontsize=30)
Label(f[1, 4], "B ", alignmode=Inside(), tellwidth=false, tellheight=false, valign=:top, halign=:right, fontsize=30)
current_figure()
