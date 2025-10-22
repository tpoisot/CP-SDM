function _softmax(p)
    w = [exp(p), exp(1 - p)]
    return w ./ sum(w)
end

function _no_softmax(p)
    w = [p, 1 - p]
    return w
end

function conformal(sdm, tr, val; α=0.05, w₊=1.0, w₋=1.0, softmax=true, kwargs...)
    model = deepcopy(sdm)

    # This is the scoring function
    class_scores = softmax ? _softmax : _no_softmax

    # We train the model on the training set
    train!(model; training=tr, kwargs...)

    # And now we start calibration
    f̂ = predict(model; threshold=false)[val]

    # The first step is to get a series of scores for the calibration
    # predictions
    s = class_scores.(f̂)

    # Because we do Mondrian prediction, we accumulate the scores in different
    # vectors
    s₊ = zeros(sum(labels(model)[val]))
    s₋ = zeros(sum(.!labels(model)[val]))

    # We also need a counter for each
    c₊ = 0
    c₋ = 0

    # And now we loop, and assign the correct conformity score, but multiplied
    # by the relative weight of each class
    for i in eachindex(val)
        c = 1 - (labels(model)[val][i] ? s[i][1] : s[i][2])
        if labels(model)[val][i]
            c₊ += 1
            s₊[c₊] = w₊ * c
        else
            c₋ += 1
            s₋[c₋] = w₋ * c
        end
    end

    # We then get the quantiles for each of the classes
    qᵢ₊ = ceil((c₊ + 1) * (1 - α)) / c₊
    qᵢ₋ = ceil((c₋ + 1) * (1 - α)) / c₋
    q₊ = quantile(s₊, qᵢ₊)
    q₋ = quantile(s₋, qᵢ₋)

    # And we return
    return (q₊, q₋)
end

function credibleclasses(ŷ, q₊, q₋; w₊=1.0, w₋=1.0, softmax=true)

    # Scoring function
    class_scores = softmax ? _softmax : _no_softmax
    
    # We get the scores
    p₊, p₋ = 1.0 .- class_scores(ŷ)

    # And now we collect the credible classes in a Set
    ℂ = Set()
    if (w₊ * p₊) <= q₊
        push!(ℂ, true)
    end
    if (w₋ * p₋) <= q₋
        push!(ℂ, false)
    end
    return ℂ
end
