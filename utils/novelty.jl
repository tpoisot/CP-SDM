function novelty(historical, projected, vars)
    μ = mean.(historical[vars])
    σ = std.(historical[vars])
    cr_historical = (historical[vars] .- μ) ./ σ
    cr_projected = (projected[vars] .- μ) ./ σ

    Δclim = similar(cr_historical[1])

    k = keys(cr_historical[1])
    vals = values.(cr_historical)

    # Thread-safe structure
    chunk_size = max(1, length(k) ÷ (5 * Threads.nthreads()))
    data_chunks = Base.Iterators.partition(k, chunk_size)

    tasks = map(data_chunks) do chunk
        Threads.@spawn begin
            for position in chunk
                diffs = [(cr_projected[i][position] .- vals[i]) .^ 2.0 for i in eachindex(vals)]
                sml_dist = findmin(sqrt.(sum(hcat(diffs...); dims=2)))
                Δclim[position] = first(sml_dist)
            end
        end
    end

    fetch.(tasks)

    return Δclim
end