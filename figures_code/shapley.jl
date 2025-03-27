# Varord
vord = sortperm(svimp; rev=true)

# Map with Shapley and contributions
f = Figure(; size=(1200, 600))
ax = Axis(f[1:2, 1]; aspect=DataAspect())
hm = heatmap!(ax, S[smimp], colormap=:diverging_bwg_20_95_c41_n256, colorrange=shaplim(S[smimp]))
for p in polygons
    lines!(ax, p, color=:grey10, linewidth=1)
end
Colorbar(
    f[1:2, 1],
    hm;
    label="Impact on prediction",
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
ax2 = Axis(f[1, 2], xlabel="Effect on prediction")
bins = LinRange(shaplim(S[smimp])..., 100)
hist!(ax2, mask(S[smimp], nodata(sure_absence, false)), color=(:orange, 0.7), bins=bins, normalization=:pdf)
hist!(ax2, mask(S[smimp], nodata(sure_presence, false)), color=(:grey80, 0.7), bins=bins, normalization=:pdf)
hist!(ax2, mask(S[smimp], nodata(unsure, false)), color=(:forestgreen, 0.7), bins=bins, normalization=:pdf)
tightlimits!(ax2)
hideydecorations!(ax2)
hidespines!(ax2, :r)
hidespines!(ax2, :l)
hidespines!(ax2, :t)

# Ticks
vnames = layers(provider)[variables(sdm)[vord]]
ax3 = Axis(f[2, 2], ylabel="Relative importance", xticks=(1:length(vord), vnames))
surea_imp = [mean(abs.(mask(ex, nodata(sure_absence, false)))) for ex in S]
uns_imp = [mean(abs.(mask(ex, nodata(unsure, false)))) for ex in S]
surep_imp = [mean(abs.(mask(ex, nodata(sure_presence, false)))) for ex in S]
scatterlines!(ax3, svimp[vord] ./ sum(svimp), color=:black, linewidth=1, linestyle=:dot)
scatterlines!(ax3, surea_imp[vord] ./ sum(surea_imp), color=:orange, label="Sure absence")
scatterlines!(ax3, uns_imp[vord] ./ sum(uns_imp), color=:grey60, label="Unsure")
scatterlines!(ax3, surep_imp[vord] ./ sum(surep_imp), color=:forestgreen, label="Sure presence")
axislegend(ax3)

Label(f[1:2, 1], "A", alignmode=Inside(), tellwidth=false, tellheight=false, valign=:top, halign=:left, fontsize=30)
Label(f[1, 2], "B", alignmode=Inside(), tellwidth=false, tellheight=false, valign=:top, halign=:left, fontsize=30)
Label(f[2, 2], " C", alignmode=Inside(), tellwidth=false, tellheight=false, valign=:bottom, halign=:left, fontsize=30)

current_figure()
