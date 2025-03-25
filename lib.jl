function _softmax(p)
    w = [exp(p), exp(1-p)]
    return w ./ sum(w)
end

function _no_softmax(p)
    w = [p, 1-p]
    return w
end

function _estimate_q(model, St, Sv; α=0.1, kwargs...)
    train!(model; training=St, kwargs...)
    f̂ = predict(model; threshold=false)[Sv]
    𝐶 = zeros(length(f̂))
    s = _no_softmax.(f̂)
    # Conformal score from the true class label
    for i in eachindex(𝐶)
        𝐶[i] = 1 - (labels(model)[Sv[i]] ? s[i][1] : s[i][2])
    end
    # Get the quantile
    n = length(Sv)
    qᵢ = ceil((n+1)*(1-α))/n
    q̂ = quantile(𝐶, qᵢ)
    return q̂
end

function credibleclasses(prediction, q)
    presence = zeros(prediction, Bool)
    absence = zeros(prediction, Bool)
    for k in keys(prediction)
        s₊, s₋ = _no_softmax(prediction[k])
        if s₊ >= (1-q)
            presence[k] = true
        end
        if s₋ >= (1-q)
            absence[k] = true
        end
    end
    return (presence, absence)
end

# Yah
