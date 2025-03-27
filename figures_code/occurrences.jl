
# Plot the figure for presence/absence
f = Figure(; size=(1000, 600))
ax = Axis(f[1:2, 1]; aspect=DataAspect())
for p in polygons
    poly!(ax, p, color=:grey95)
    lines!(ax, p, color=:grey10)
end
scatter!(ax, presencelayer, color=:white, strokecolor=:forestgreen, strokewidth=2)
scatter!(ax, bgpoints, color=:grey30, markersize=4)
hidespines!(ax)
hidedecorations!(ax)
C = [ConfusionMatrix(predict(sdm; threshold=false), labels(sdm), t) for t in LinRange(0.0, 1.0, 150)]
ax2 = Axis(f[1, 2], aspect=DataAspect(), xlabel="False positive rate", ylabel="True positive rate")
lines!(ax2, [0, 1], [0, 1], color=:grey50, linestyle=:dot)
lines!(ax2, fpr.(C), tpr.(C), color=:black)
ax3 = Axis(f[2, 2], aspect=DataAspect(), xlabel="Precision", ylabel="Recall")
hlines!(ax3, [SDeMo.recall(noskill(sdm))], color=:grey50, linestyle=:dot)
lines!(ax3, SDeMo.precision.(C), SDeMo.recall.(C), color=:black)
ylims!(ax2, 0, 1)
ylims!(ax3, 0, 1)
xlims!(ax2, 0, 1)
xlims!(ax3, 0, 1)

Label(f[1:2, 1], "A", alignmode=Inside(), tellwidth=false, tellheight=false, valign=:top, halign=:left, fontsize=30)
Label(f[1, 2], "B", alignmode=Inside(), tellwidth=false, tellheight=false, valign=:top, halign=:left, fontsize=30)
Label(f[2, 2], "C", alignmode=Inside(), tellwidth=false, tellheight=false, valign=:top, halign=:left, fontsize=30)

current_figure()
