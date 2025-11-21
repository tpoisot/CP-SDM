idx = findfirst(==(3), variables(sdm))

xy = [partialresponse(sdm, variables(sdm)[idx]; inflated=true, threshold=false) for i in 1:1000]
partialscores = hcat(last.(xy)...)
outcome = zeros(Int64, size(partialscores))
for i in eachindex(outcome)
    cc = credibleclasses(partialscores[i], q₊, q₋)
    if length(cc) == 2
        outcome[i] = 1
    else
        outcome[i] = (true in cc) ? 2 : 0
    end
end

f = Figure()
ax = Axis(f[1,1], xlabel="BIO$(variables(sdm)[idx])", ylabel="Proportion of outcomes")
lines!(ax, xy[1][1], vec(mean(outcome.==2; dims=2)), color=:forestgreen, label="Sure presence")
lines!(ax, xy[1][1], vec(mean(outcome.==1; dims=2)), color=:grey50, label="Unsure")
lines!(ax, xy[1][1], vec(mean(outcome.==0; dims=2)), color=:orange, label="Sure absence")
axislegend(ax, nbanks=3, position=:lt)
f