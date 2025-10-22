# Mondrian CP

model = sdm
fold = holdout(model)
kwargs = (;)

# We train the model
train!(model; training=fold[1], kwargs...)

α = 0.1

# We now get the SCORES for each version
f̂ = predict(model; threshold=false)[fold[2]]
C = zeros(length(f̂))
s = _softmax.(f̂)

s₊ = zeros(sum(labels(model)[fold[2]]))
s₋ = zeros(sum(.!labels(model)[fold[2]]))

c₊ = 0
c₋ = 0

# TODO weights for the different classes to show different priorities

for i in eachindex(C)
    c = 1 - (labels(model)[fold[2][i]] ? s[i][1] : s[i][2])
    if labels(model)[fold[2]][i]
        c₊ += 1
        s₊[c₊] = c
    else
        c₋ += 1
        s₋[c₋] = c
    end
end
n = length(fold[2])
qᵢ = ceil((n + 1) * (1 - α)) / n
q₊ = quantile(s₊, qᵢ)
q₋ = quantile(s₋, qᵢ)

# TODO figure of the q+ and q- when changing the coverage threshold to demo Mondrian

U = predict(model, L; threshold=false)

function credible(y, q)
    p₊, p₋ = 1.0 .- _softmax(y)
    q₊, q₋ = q
    C = Set()
    if p₊ <= q₊
        push!(C, true)
    end
    if p₋ <= q₋
        push!(C, false)
    end
    return C
end

cp = [credible(y, (q₊, q₋)) for y in predict(model; threshold=false)]

uncertain = (x -> length(credible(x, (q₊, q₋)))).(U)
heatmap(predict(model, L), colormap=[:white, :darkgreen])
heatmap!(uncertain, colormap=[:transparent, :grey70])
contour!(predict(model, L), color=:darkgreen)
lines!(landmass, color=:black)
scatter!(presencelayer, color=:lime, markersize=4)
current_figure()