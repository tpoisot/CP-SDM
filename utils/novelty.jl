"""
Measure the novelty based on two matrices with the correct observation. The
reference is always given as the data from which the mean and standard deviation
are applied. The flip keyword is used to decide which dataset is used as a the
baseline. If flip = true, the target is used as the reference, i.e. we get one
output for every point in the target dataset. Otherwise, we get one output for
every point in the reference dataset (default).
"""
function _novelty_source_ref(reference, target; flip=false)
    μ = vec(mean(reference; dims=2))
    σ = vec(mean(reference; dims=2))
    R = (reference .- μ) ./ σ
    T = (target .- μ) ./ σ

    # We create an array to store the minimum distances
    idx = 1:(flip ? size(T, 2) : size(R, 2))
    D = zeros(Float32, length(idx))

    # And then we chunk for parallel processing
    chunk_size = max(1, length(idx) ÷ (5 * Threads.nthreads()))
    data_chunks = Base.Iterators.partition(idx, chunk_size)

    # We loop (threaded)
    tasks = map(data_chunks) do chunk
        Threads.@spawn begin
            for i in chunk
                diffs = flip ? (T[:, i] .- R) : (R[:, i] .- T)
                diffs .*= diffs
                rmse = sqrt.(vec(sum(diffs; dims=2)))
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
    nov = _novelty_source_ref(reference, target; kwargs...)
    return SpeciesDistributionToolkit.burnin(dist, nov)
end

function novelty(layers::Vector{<:SDMLayer}, model::SDM; kwargs...)
    reference = SDT._X_from_layers(layers[variables(model)])
    target = instance(model, :)
    dist = similar(first(layers))
    nov = _novelty_source_ref(reference, target; kwargs...)
    return SpeciesDistributionToolkit.burnin(dist, nov)
end