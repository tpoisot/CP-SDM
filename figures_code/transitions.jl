# Generate a bivariate palette
function _palette(; low=colorant"#e8e8e8", high=colorant"#120fe3", breaks=3)
    breakpoints = LinRange(0.0, 1.0, breaks)
    return ColorSchemes.weighted_color_mean.(breakpoints, high, low)
end

# Get the palettes
nbreaks = 3

# 
low = colorant"#f8f8f8"
h1 = colorant"#73ae80"
h2 = colorant"#6c83b5"

colormap1 = _palette(; low=low, high=h1, breaks=nbreaks)
colormap2 = _palette(; low=low, high=h2, breaks=nbreaks)
colormatrix = [ColorBlendModes.blend.(c1, c2; mode=BlendMultiply) for c1 in colormap1, c2 in colormap2]


# Discrete maps
m1 = current_uncertainty .+ 1
m2 = future_uncertainty .+ 1
category = similar(m1)
for i in eachindex(category)
    category[i] = LinearIndices(colormatrix)[m1[i], m2[i]]
end

f = Figure(; size=(1200, 600))
ax1 = Axis(f[1, 1]; aspect=DataAspect())
ax2 = Axis(f[1, 2]; aspect=0.8)
heatmap!(ax1, category, colormap=vec(colormatrix))
lines!(ax1, landmass, color=:black)
f

total_area = sum(cell_surface)
current_bars = [sum(mask(cell_surface, nodata(current_uncertainty .== i, false))) for i in 0:2]
future_bars = [sum(mask(cell_surface, nodata(future_uncertainty .== i, false))) for i in 0:2]

# Get the transition matrix
trs = zeros(Float64, 3, 3)
for cur in [0, 1, 2]
    this_cur = current_uncertainty .== cur
    for fut in [0, 1, 2]
        this_fut = future_uncertainty .== fut
        this_tr_area = sum(mask(cell_surface, nodata(this_cur .& this_fut, false)))
        trs[cur+1, fut+1] = this_tr_area
    end
end

# Transition matrix (normalized by extent)

T = trs ./ total_area

future_sum = sum(T, dims=1) # Future data
current_sum = sum(T, dims=2) # Current data

Tf = T ./ future_sum # row sums to 1
Tc = T ./ current_sum # col sums to 1

current_boxes = Tuple{Float64, Float64}[]
future_boxes = Tuple{Float64, Float64}[]

for i in eachindex(future_sum)
    spacer = (i-1)*0.1
    start_at = i == 1 ? 0.0 : sum(future_sum[1:(i-1)]) + spacer
    stop_at = sum(future_sum[1:i]) + spacer
    push!(future_boxes, (start_at, stop_at))
end
for i in eachindex(current_sum)
    spacer = (i-1)*0.1
    start_at = i == 1 ? 0.0 : sum(current_sum[1:(i-1)]) + spacer
    stop_at = sum(current_sum[1:i]) + spacer
    push!(current_boxes, (start_at, stop_at))
end

for i in axes(T, 1)
    cb = current_boxes[i]
    ch = cb[2] - cb[1] # Height of current box
    for j in axes(T, 2)
        fb = future_boxes[j]
        fh = fb[2] - fb[1] # Height of future box
        # Starting point from the current box
        if Tc[i,j] > 0.0
            csr = j == 1 ? 0.0 : sum(Tc[i,1:(j-1)]) # Current start (relative)
            cer = sum(Tc[i,1:j]) # Current end (relative)
            c_start = cb[1] + ch * csr
            c_stop = cb[1] + ch * cer
            fsr = i == 1 ? 0.0 : sum(Tf[1:(i-1),j]) # Current start (relative)
            fer = sum(Tf[1:i,j]) # Current end (relative)
            f_start = fb[1] + fh * fsr
            f_stop = fb[1] + fh * fer
            # Calculate the band
            quad_easing(x) = x < 0.5 ? 2 * (x^2) : 1 - (-2 * x + 2)^2 / 2
            vshift = quad_easing.(LinRange(0.0, 1.0, 50))
            Δb = f_start - c_start
            Δt = f_stop - c_stop
            xt = LinRange(0.05, 0.95, length(vshift))
            yb = c_start .+ Δb .* vshift
            yt = c_stop .+ Δt .* vshift
            band!(ax2, xt, yt, yb, color=colormatrix[i,j])
        end
    end
end

for i in eachindex(current_boxes)
    cb = current_boxes[i]
    fb = future_boxes[i]
    poly!(ax2, Point2f[(0, cb[1]), (0.05, cb[1]), (0.05, cb[2]), (0, cb[2])], color=colormap1[i], strokecolor=:black, strokewidth=1)
    poly!(ax2, Point2f[(0.95, fb[1]), (1.0,fb[1]), (1.0, fb[2]), (0.95, fb[2])], color=colormap2[i], strokecolor=:black, strokewidth=1)
end

for (i, l) in enumerate(["Absent", "Unsure", "Present"])
    text!(ax2, [(-0.08, mean(current_boxes[i]))], text = l, rotation=π/2, align=(:center, :top))
    text!(ax2, [(1.08, mean(future_boxes[i]))], text = l, rotation=π/2, align=(:center, :bottom))
end

hidespines!(ax1)
hidedecorations!(ax1)
hidespines!(ax2)
hidedecorations!(ax2)

f