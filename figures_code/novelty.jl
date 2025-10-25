tr_conserved = nodata(sure_presence .& ft_sure_presence, false)
tr_maybegain =  nodata(sure_absence .& ft_unsure, false)
tr_maybeloss = nodata(sure_presence .& ft_unsure, false)
tr_ambig = nodata(unsure .& ft_unsure, false)
tr_suregain = nodata((sure_absence .| unsure) .& ft_sure_presence, false)
tr_sureloss = nodata((sure_presence .| unsure) .& ft_sure_absence, false)

# Novelty map + histograms
f = Figure(; size=(1200, 600))
ax = Axis(f[1:3, 1:2]; aspect=DataAspect())
hm = heatmap!(ax, lost_climates, colormap=:Oranges, colorscale=sqrt)
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
ax_np = Axis(f[1, 3]; ylabel="Future presence")
ax_nu = Axis(f[2, 3]; ylabel="Future uncertain")
ax_na = Axis(f[3, 3]; ylabel="Future absence", xlabel="Novel climate index")
ax_lp = Axis(f[1, 4]; ylabel="Future presence")
ax_lu = Axis(f[2, 4]; ylabel="Future uncertain")
ax_la = Axis(f[3, 4]; ylabel="Future absence", xlabel="Lost climate index")

coolbins = LinRange(0.1, 1.2, 80)

# Novel climate
hist!(ax_np, mask(novel_climates, nodata(ft_sure_presence, false)), bins=coolbins, color=:darkgreen, strokecolor=:black, strokewidth=0.5)
hist!(ax_np, mask(novel_climates, nodata(ft_sure_presence .& (unsure .| sure_absence), false)), bins=coolbins, color=:grey50, strokecolor=:black, strokewidth=0.5)
hist!(ax_np, mask(novel_climates, nodata(ft_sure_presence .& sure_absence, false)), bins=coolbins, color=:white, strokecolor=:black, strokewidth=0.5)

hist!(ax_nu, mask(novel_climates, nodata(ft_unsure, false)), bins=coolbins, color=:darkgreen, strokecolor=:black, strokewidth=0.5)
hist!(ax_nu, mask(novel_climates, nodata(ft_unsure .& (unsure .| sure_absence), false)), bins=coolbins, color=:grey50, strokecolor=:black, strokewidth=0.5)
hist!(ax_nu, mask(novel_climates, nodata(ft_unsure .& sure_absence, false)), bins=coolbins, color=:white, strokecolor=:black, strokewidth=0.5)

hist!(ax_na, mask(novel_climates, nodata(ft_sure_absence, false)), bins=coolbins, color=:darkgreen, strokecolor=:black, strokewidth=0.5)
hist!(ax_na, mask(novel_climates, nodata(ft_sure_absence .& (unsure .| sure_absence), false)), bins=coolbins, color=:grey50, strokecolor=:black, strokewidth=0.5)
hist!(ax_na, mask(novel_climates, nodata(ft_sure_absence .& sure_absence, false)), bins=coolbins, color=:white, strokecolor=:black, strokewidth=0.5)

# Lost climates
hist!(ax_lp, mask(lost_climates, nodata(ft_sure_presence, false)), bins=coolbins, color=:darkgreen, strokecolor=:black, strokewidth=0.5)
hist!(ax_lp, mask(lost_climates, nodata(ft_sure_presence .& (unsure .| sure_absence), false)), bins=coolbins, color=:grey50, strokecolor=:black, strokewidth=0.5)
hist!(ax_lp, mask(lost_climates, nodata(ft_sure_presence .& sure_absence, false)), bins=coolbins, color=:white, strokecolor=:black, strokewidth=0.5)

hist!(ax_lu, mask(lost_climates, nodata(ft_unsure, false)), bins=coolbins, color=:darkgreen, strokecolor=:black, strokewidth=0.5)
hist!(ax_lu, mask(lost_climates, nodata(ft_unsure .& (unsure .| sure_absence), false)), bins=coolbins, color=:grey50, strokecolor=:black, strokewidth=0.5)
hist!(ax_lu, mask(lost_climates, nodata(ft_unsure .& sure_absence, false)), bins=coolbins, color=:white, strokecolor=:black, strokewidth=0.5)

hist!(ax_la, mask(lost_climates, nodata(ft_sure_absence, false)), bins=coolbins, color=:darkgreen, strokecolor=:black, strokewidth=0.5)
hist!(ax_la, mask(lost_climates, nodata(ft_sure_absence .& (unsure .| sure_absence), false)), bins=coolbins, color=:grey50, strokecolor=:black, strokewidth=0.5)
hist!(ax_la, mask(lost_climates, nodata(ft_sure_absence .& sure_absence, false)), bins=coolbins, color=:white, strokecolor=:black, strokewidth=0.5)

for ax in [ax_np, ax_nu, ax_na, ax_lp, ax_lu, ax_la]
    xlims!(ax, extrema(coolbins))
    hideydecorations!(ax, label=false)
    tightlimits!(ax)
end

Label(f[1:3, 1:2], "A", alignmode=Inside(), tellwidth=false, tellheight=false, valign=:top, halign=:left, fontsize=30)
Label(f[1, 4], "B ", alignmode=Inside(), tellwidth=false, tellheight=false, valign=:top, halign=:right, fontsize=30)
current_figure()
