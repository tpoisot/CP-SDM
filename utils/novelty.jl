function _novelty_source_ref(baseline, comparison)
    μ = vec(mean(baseline; dims=2))
    σ = vec(mean(baseline; dims=2))
    B = (baseline .- μ) ./ σ
    C = (comparison .- μ) ./ σ

    # We create an array to store the minimum distances
    idx = 1:size(B, 2)
    D = zeros(Float32, length(idx))

    # And then we chunk for parallel processing
    chunk_size = max(1, length(idx) ÷ (5 * Threads.nthreads()))
    data_chunks = Base.Iterators.partition(idx, chunk_size)

    # We loop (threaded)
    tasks = map(data_chunks) do chunk
        Threads.@spawn begin
            for i in chunk
                Δ = (B[:,i] .- C) .^2.0
                rmse = sqrt.(vec(sum(Δ; dims=1)))
                D[i] = first(findmin(rmse))
            end
        end
    end

    fetch.(tasks)
    return D
end

function novelty(historical::Vector{SDMLayer}, future::Vector{SDMLayer}; kwargs...)
    reference = SDT._X_from_layers(historical)
    target = SDT._X_from_layers(future)
    dist = similar(first(historical))
    return SpeciesDistributionToolkit.burnin(dist, _novelty_source_ref(reference, target; kwargs...))
end

function novelty(model::SDM, layers::Vector{<:SDMLayer}; kwargs...)
    reference = instance(model, :)
    target = SDT._X_from_layers(layers[variables(model)])
    dist = similar(first(layers))
    nov = _novelty_source_ref(target, reference; kwargs...)
    return SpeciesDistributionToolkit.burnin(dist, nov)
end

function novelty(layers::Vector{<:SDMLayer}, model::SDM; kwargs...)
    reference = SDT._X_from_layers(layers[variables(model)])
    target = instance(model, :)
    dist = similar(first(layers))
    nov = _novelty_source_ref(reference, target; kwargs...)
    return SpeciesDistributionToolkit.burnin(dist, nov)
end