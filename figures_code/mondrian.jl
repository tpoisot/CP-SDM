# Risk level at which an area becomes certain
iscertain(p, q1, q2) = length(credibleclasses(p, q1, q2)) == 1
uncmap = [(y -> iscertain(y, q...)).(predict(sdm, L; threshold=false)) for q in qs]
function uncindex(v, rl)
    u = findall(v)
    return isempty(u) ? NaN : maximum(rl[u])
end
uncmosaic = mosaic(v -> uncindex(v, risk_levels), uncmap)

f = Figure(; size=(1200, 600))
ax = Axis(f[1:2, 1]; aspect=DataAspect())
hm = heatmap!(ax, uncmosaic, colormap=:davos)
Colorbar(
    f[1:2, 1],
    hm;
    label="Risk level",
    alignmode=Inside(),
    height=Relative(0.5),
    flipaxis=false,
    valign=:bottom,
    halign=:right,
    tellheight=false,
    tellwidth=false,
    vertical=true,
)
lines!(ax, landmass, color=:black)
hidespines!(ax)
hidedecorations!(ax)

ax2 = Axis(f[1, 2], xlabel="Risk level", ylabel="Threshold for the positive class")
ax3 = Axis(f[2, 2], xlabel="Risk level", ylabel="Threshold for the negative class")
x, m, s = _agr(risk_levels, first.(qs))
errorbars!(ax2, x, m, s, whiskerwidth=10, color=:darkgreen, depth_shift=-1.0)
scatter!(ax2, x, m, strokecolor=:darkgreen, strokewidth=3, color=:white, label="Presence.")

x, m, s = _agr(risk_levels, last.(qs))
errorbars!(ax3, x, m, s, whiskerwidth=10, color=:grey50, depth_shift=-1.0)
scatter!(ax3, x, m, strokecolor=:grey50, strokewidth=3, color=:white, label="Absence.")

ylims!(ax2, 0.25, 0.8)
ylims!(ax3, 0.25, 0.8)

current_figure()
