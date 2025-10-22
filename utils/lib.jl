function _softmax(p)
    w = [exp(p), exp(1 - p)]
    return w ./ sum(w)
end

function _no_softmax(p)
    w = [p, 1 - p]
    return w
end

function _estimate_q(model, St, Sv; α=0.1, kwargs...)
    train!(model; training=St, kwargs...)
    f̂ = predict(model; threshold=false)[Sv]
    𝐶 = zeros(length(f̂))
    s = _softmax.(f̂)
    # Conformal score from the true class label
    for i in eachindex(𝐶)
        𝐶[i] = 1 - (labels(model)[Sv[i]] ? s[i][1] : s[i][2])
    end
    # Get the quantile
    n = length(Sv)
    qᵢ = ceil((n + 1) * (1 - α)) / n
    q̂ = quantile(𝐶, qᵢ)
    return q̂
end

# TODO: need to make an additional function for adaptive prediction sets, and
# while at it, might refactor to do different types of predictions in a more
# formal way

function credibleclasses(prediction::SDMLayer, q)
    presence = zeros(prediction, Bool)
    absence = zeros(prediction, Bool)
    for k in keys(prediction)
        ℂ = credibleclasses(prediction[k], q)
        if true in ℂ
            presence[k] = true
        end
        if false in ℂ
            absence[k] = true
        end
    end
    return (presence, absence)
end

function credibleclasses(prediction::Number, q)
    s₊, s₋ = _softmax(prediction)
    out = Bool[]
    if s₊ >= (1 - q)
        push!(out, true)
    end
    if s₋ >= (1 - q)
        push!(out, false)
    end
    return Set(out)
end
