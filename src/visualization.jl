using Plots, CairoMakie, Statistics

"""
    split_bits(dec::Int, n::Int)

Splits a decimal index into two coordinates based on its bitstring halves.
Returns (x, y) coordinates.
"""
function split_bits(dec::Int, n::Int)
    n_left = div(n, 2)
    n_right = n - n_left
    
    # Extract the left half from the lower-order bits.
    x = dec & ((1 << n_left) - 1)
    # Extract the right half from the upper-order bits.
    y = dec >> n_left
    
    return (x, y)
end

"""
    plot_hbm(landscape::Landscape, epsilon::Float64, local_optima::Vector{Int}=Int[]; save_path::String="", return_fitness_range::Bool=false)

Plots the Hinged Bitstring Map for the landscape.
"""
function plot_hbm(
    landscape::Landscape,
    epsilon::Float64,
    local_optima::Vector{Int}=Int[];
    save_path::String="",
    return_fitness_range::Bool=false
)
    n = landscape.n
    if n > 20
        error("HBM plotting is disabled for n > 20 to avoid exhaustive visualization on huge landscapes")
    end
    num_combinations = 2^n
    
    xs = Int[]
    ys = Int[]
    fs = Float64[]
    
    start_dec = landscape.is_synthetic ? 0 : 1
    end_dec = num_combinations - 1
    
    for i in start_dec:end_dec
        x, y = split_bits(i, n)
        f = evaluate_fitness(landscape, i, epsilon)
        push!(xs, x)
        push!(ys, y)
        push!(fs, f)
    end
    
    # Keep the legend above the plot and leave extra room for the title.
    p = Plots.scatter(xs, ys, zcolor=fs, markerstrokewidth=0, markersize=3, 
                title="HBM: $(landscape.name) (ε=$epsilon)",
                titlefontsize=11,
                xlabel="Left bits", ylabel="Right bits", 
                colorbar_title="Fitness",
                label="Search Space",
                legend=(0.85, 1.12),
                legendfontsize=7,
                background_color_legend=RGBA(1,1,1,0.6),
                top_margin=14Plots.mm, bottom_margin=12Plots.mm,
                left_margin=12Plots.mm, right_margin=12Plots.mm,
                aspect_ratio=:equal, size=(1100, 800))
    
    best_f = maximum(fs)
    
    if !isempty(local_optima)
        oxs = Int[]
        oys = Int[]
        for o in local_optima
            x, y = split_bits(o, n)
            push!(oxs, x)
            push!(oys, y)
        end
        Plots.scatter!(p, oxs, oys, markershape=:circle, markercolor=:transparent, 
                 markerstrokecolor=:red, markerstrokewidth=1, markersize=4, label="Local Optima")
    end
    
    global_indices = findall(f -> f ≈ best_f, fs)
    gxs = xs[global_indices]
    gys = ys[global_indices]
    Plots.scatter!(p, gxs, gys, markershape=:star5, markercolor=:gold, 
             markersize=8, label="Global Optima")
    
    if save_path != ""
        Plots.savefig(p, save_path)
    end

    if return_fitness_range
        return p, minimum(fs), maximum(fs)
    end
    return p
end

"""
    plot_convergence(history::Vector{Vector{Float64}}; title::String="", save_path::String="")
"""
function plot_convergence(history::Vector{Vector{Float64}}; title::String="", save_path::String="")
    mins = [minimum(h) for h in history]
    avgs = [mean(h) for h in history]
    maxs = [maximum(h) for h in history]
    plot_title = isempty(strip(title)) ? "Convergence" : title
    
    p = Plots.plot(maxs, label="Max", title=plot_title, titlefontsize=11,
                   xlabel="Generation", ylabel="Fitness", 
                   legend=(0.9, 1.12), legendfontsize=8,
                   top_margin=14Plots.mm, bottom_margin=12Plots.mm,
                   left_margin=15Plots.mm, right_margin=15Plots.mm,
                   size=(900, 600))
    Plots.plot!(p, avgs, label="Avg")
    Plots.plot!(p, mins, label="Min")
    
    if save_path != ""
        Plots.savefig(p, save_path)
    end
    
    return p
end

"""
    plot_convergence_aggregated(all_histories; title::String="", save_path::String="")
"""
function plot_convergence_aggregated(all_histories; title::String="", save_path::String="")
    if isempty(all_histories)
        error("Cannot plot aggregated convergence: all_histories is empty")
    end

    n_gens = minimum(length.(all_histories))
    run_best_matrix = zeros(Float64, length(all_histories), n_gens)

    for (r_idx, run_hist) in enumerate(all_histories)
        for gen in 1:n_gens
            run_best_matrix[r_idx, gen] = maximum(run_hist[gen])
        end
    end

    mean_best = vec(mean(run_best_matrix, dims=1))
    min_best = vec(minimum(run_best_matrix, dims=1))
    max_best = vec(maximum(run_best_matrix, dims=1))
    plot_title = isempty(strip(title)) ? "Convergence (Aggregated)" : title

    p = Plots.plot(
        mean_best,
        label="Mean Best",
        title=plot_title,
        titlefontsize=11,
        xlabel="Generation",
        ylabel="Best Fitness",
        legend=(0.9, 1.12),
        legendfontsize=8,
        top_margin=14Plots.mm, bottom_margin=12Plots.mm,
        left_margin=15Plots.mm, right_margin=15Plots.mm,
        size=(900, 600)
    )
    Plots.plot!(p, min_best, label="Min Best")
    Plots.plot!(p, max_best, label="Max Best")

    if save_path != ""
        Plots.savefig(p, save_path)
    end

    return p
end

function run_entropy_over_time(population_history::Vector{Vector{Int}}, n::Int)
    entropies = Float64[]
    for pop in population_history
        coords = [split_bits(ind, n) for ind in pop]
        counts = countmap(coords)
        probs = values(counts) ./ length(pop)
        push!(entropies, -sum(p * log2(p) for p in probs))
    end
    return entropies
end

function unique_states_over_time(population_history::Vector{Vector{Int}})
    return [length(unique(pop)) for pop in population_history]
end

function cumulative_unique_states_over_time(population_history::Vector{Vector{Int}})
    seen = Set{Int}()
    cum = Int[]
    for pop in population_history
        union!(seen, pop)
        push!(cum, length(seen))
    end
    return cum
end

function pareto_front_size_over_time(population_history::Vector{Vector{Int}}, landscape::Landscape)
    sizes = Int[]
    for pop in population_history
        objs = [evaluate_multiobjective(landscape, ind) for ind in pop]
        n_pop = length(pop)
        is_nondominated = trues(n_pop)
        for i in 1:n_pop
            for j in 1:n_pop
                if i != j && all(objs[j] .<= objs[i]) && any(objs[j] .< objs[i])
                    is_nondominated[i] = false
                    break
                end
            end
        end
        push!(sizes, count(is_nondominated))
    end
    return sizes
end

function mean_series_over_runs(all_runs, metric_fn)
    if isempty(all_runs)
        return Float64[]
    end
    n_gens = minimum(length.(all_runs))
    acc = zeros(Float64, n_gens)
    for run in all_runs
        vals = metric_fn(run)[1:n_gens]
        acc .+= Float64.(vals)
    end
    return acc ./ length(all_runs)
end

function segmented_path_xy(xs::AbstractVector{<:Integer}, ys::AbstractVector{<:Integer}, keep_segment::AbstractVector{Bool})
    if length(xs) <= 1 || length(keep_segment) != length(xs) - 1
        return Float64[], Float64[]
    end
    sx = Float64[]
    sy = Float64[]
    for i in 1:length(keep_segment)
        if keep_segment[i]
            push!(sx, xs[i], xs[i+1], NaN)
            push!(sy, ys[i], ys[i+1], NaN)
        end
    end
    return sx, sy
end

function centered_clamped_limits(center::Float64, span::Float64, lo::Float64, hi::Float64)
    usable_span = min(span, hi - lo)
    lower = center - usable_span / 2
    upper = center + usable_span / 2

    if lower < lo
        upper += lo - lower
        lower = lo
    end
    if upper > hi
        lower -= upper - hi
        upper = hi
    end

    return (max(lo, lower), min(hi, upper))
end

function trajectory_zoom_limits(
    xs::AbstractVector{<:Real},
    ys::AbstractVector{<:Real},
    n::Int;
    pad_frac::Float64=0.12,
    min_frac::Float64=0.03,
    edge_pad::Float64=0.65
)
    n_left = div(n, 2)
    n_right = n - n_left

    x_lo, x_hi = -edge_pad, Float64((1 << n_left) - 1) + edge_pad
    y_lo, y_hi = -edge_pad, Float64((1 << n_right) - 1) + edge_pad

    x_vals = Float64.(xs)
    y_vals = Float64.(ys)
    x_min, x_max = minimum(x_vals), maximum(x_vals)
    y_min, y_max = minimum(y_vals), maximum(y_vals)

    x_domain_span = max((Float64((1 << n_left) - 1)) - 0.0, 1.0)
    y_domain_span = max((Float64((1 << n_right) - 1)) - 0.0, 1.0)
    x_local_span = max(x_max - x_min, min_frac * x_domain_span, 2.0)
    y_local_span = max(y_max - y_min, min_frac * y_domain_span, 2.0)
    x_axis_span = x_local_span * (1 + 2 * pad_frac)
    y_axis_span = y_local_span * (1 + 2 * pad_frac)
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)

    return (
        centered_clamped_limits(x_center, x_axis_span, x_lo, x_hi),
        centered_clamped_limits(y_center, y_axis_span, y_lo, y_hi)
    )
end

function nice_tick_step(span::Float64; target_intervals::Int=6)
    if span <= 0
        return 1.0
    end
    raw = span / target_intervals
    magnitude = 10.0 ^ floor(log10(raw))
    normalized = raw / magnitude
    base = normalized <= 1 ? 1.0 : normalized <= 2 ? 2.0 : normalized <= 5 ? 5.0 : 10.0
    return max(1.0, base * magnitude)
end

function adaptive_axis_ticks(lims::Tuple{Float64, Float64})
    lo, hi = lims
    span = hi - lo
    step = nice_tick_step(span)
    start_tick = ceil(Int, lo / step) * step
    end_tick = floor(Int, hi / step) * step
    ticks = collect(start_tick:step:end_tick)
    return isempty(ticks) ? [round(lo; digits=0), round(hi; digits=0)] : ticks
end

function active_bit_distribution_over_time(population_history::Vector{Vector{Int}}, n::Int)
    n_gens = length(population_history)
    counts = zeros(Int, n + 1, n_gens)
    for (gen_idx, pop) in enumerate(population_history)
        for ind in pop
            counts[count_ones(ind) + 1, gen_idx] += 1
        end
    end
    return counts
end

function tiny_arrow_vectors(
    xs::AbstractVector{<:Real},
    ys::AbstractVector{<:Real};
    scale::Float64=0.025,
    anchor::Float64=0.34
)
    ax = Float64[]
    ay = Float64[]
    au = Float64[]
    av = Float64[]
    for i in 1:(length(xs) - 1)
        dx = Float64(xs[i + 1] - xs[i])
        dy = Float64(ys[i + 1] - ys[i])
        seg_len = hypot(dx, dy)
        if seg_len <= 1e-12
            continue
        end
        u = dx * scale
        v = dy * scale
        push!(ax, Float64(xs[i]) + anchor * dx - 0.5 * u)
        push!(ay, Float64(ys[i]) + anchor * dy - 0.5 * v)
        push!(au, u)
        push!(av, v)
    end
    return ax, ay, au, av
end

function add_tiny_trajectory_arrows!(p, xs::AbstractVector{<:Real}, ys::AbstractVector{<:Real}; color=RGB(0.0, 0.48, 0.48))
    if length(xs) <= 1
        return p
    end
    ax, ay, au, av = tiny_arrow_vectors(xs, ys)
    if isempty(ax)
        return p
    end
    Plots.quiver!(
        p,
        ax,
        ay,
        quiver=(au, av),
        color=color,
        linewidth=0.08,
        linealpha=0.8,
        label=""
    )
    return p
end

function add_trajectory_points!(p, xs::AbstractVector{<:Real}, ys::AbstractVector{<:Real})
    if isempty(xs)
        return p
    end
    Plots.scatter!(
        p,
        xs,
        ys,
        markercolor=RGBA(0.0, 1.0, 1.0, 0.9),
        markerstrokecolor=RGB(0.0, 0.48, 0.48),
        markerstrokewidth=0.2,
        markersize=2.6,
        label=""
    )
    return p
end

function write_visit_trace(
    path::String,
    landscape::Landscape,
    epsilon::Float64,
    population_history::Vector{Vector{Int}};
    algo_name::String="",
    selected_run_index::Int=1,
    run_selection::String="best"
)
    mkpath(dirname(path))
    algo_label = isempty(strip(algo_name)) ? "Algorithm" : algo_name
    seen = Set{Int}()
    best_so_far_ind = -1
    best_so_far_fit = -Inf

    open(path, "w") do io
        println(io, "# Visit Trace")
        println(io, "landscape=$(landscape.name)")
        println(io, "algorithm=$(algo_label)")
        println(io, "epsilon=$(epsilon)")
        println(io, "selected_run_index=$(selected_run_index)")
        println(io, "run_selection=$(run_selection)")
        println(io, "")

        for (gen_idx, pop) in enumerate(population_history)
            fitnesses = [evaluate_fitness(landscape, ind, epsilon) for ind in pop]
            best_idx = argmax(fitnesses)
            current_best_ind = pop[best_idx]
            current_best_fit = fitnesses[best_idx]
            if current_best_fit > best_so_far_fit
                best_so_far_fit = current_best_fit
                best_so_far_ind = current_best_ind
            end

            unique_pop = sort(unique(pop))
            sample_count = min(8, length(unique_pop))
            sample_positions = unique_pop[1:sample_count]
            sample_suffix = length(unique_pop) > sample_count ? ",..." : ""
            union!(seen, pop)

            println(
                io,
                "[gen $(lpad(gen_idx, 3, '0'))] current_best=$(current_best_ind) " *
                "current_best_active_bits=$(count_ones(current_best_ind)) " *
                "current_best_fitness=$(round(current_best_fit; digits=6)) " *
                "best_so_far=$(best_so_far_ind) " *
                "best_so_far_fitness=$(round(best_so_far_fit; digits=6)) " *
                "unique_positions_in_gen=$(length(unique_pop)) " *
                "cumulative_unique_positions=$(length(seen)) " *
                "sample_positions=[" * join(sample_positions, ",") * sample_suffix * "]"
            )
        end
    end

    return path
end

"""
    plot_behavior(landscape::Landscape, epsilon::Float64, population_history::Vector{Vector{Int}};
                  save_path::String="", all_population_histories=nothing, algo_name::String="")
"""
function plot_behavior(
    landscape::Landscape,
    epsilon::Float64,
    population_history::Vector{Vector{Int}};
    save_path::String="",
    all_population_histories=nothing,
    algo_name::String="",
    use_landscape_map::Bool=true
)
    n = landscape.n
    algo_label = isempty(strip(algo_name)) ? "Algorithm" : algo_name

    # Panel 1: population diversity over time.
    entropies = run_entropy_over_time(population_history, n)
    p1 = Plots.plot(
        entropies,
        title="Diversity: $(algo_label) on $(landscape.name)",
        titlefontsize=11,
        xlabel="Generation", ylabel="Entropy", label="Entropy",
        legend=(0.9, 1.12), legendfontsize=7,
        top_margin=14Plots.mm, bottom_margin=12Plots.mm,
        left_margin=15Plots.mm, right_margin=15Plots.mm
    )

    # Panel 2: trajectory of the best individual.
    best_trajectory = Int[]
    best_fitness_over_time = Float64[]
    best_nones_over_time = Int[]
    for pop in population_history
        fitnesses = [evaluate_fitness(landscape, ind, epsilon) for ind in pop]
        best_idx = argmax(fitnesses)
        push!(best_trajectory, pop[best_idx])
        push!(best_fitness_over_time, fitnesses[best_idx])
        push!(best_nones_over_time, count_ones(pop[best_idx]))
    end
    
    if n <= 20 && use_landscape_map
        p2, bg_fmin, bg_fmax = plot_hbm(landscape, epsilon; save_path="", return_fitness_range=true)
        txs = [split_bits(ind, n)[1] for ind in best_trajectory]
        tys = [split_bits(ind, n)[2] for ind in best_trajectory]
        xlims_zoom, ylims_zoom = trajectory_zoom_limits(txs, tys, n)
        xticks_zoom = adaptive_axis_ticks(xlims_zoom)
        yticks_zoom = adaptive_axis_ticks(ylims_zoom)
        
        Plots.plot!(p2, txs, tys, color=:cyan, linewidth=0.9, linealpha=0.8, label="Trajectory")
        add_trajectory_points!(p2, txs, tys)
        add_tiny_trajectory_arrows!(p2, txs, tys)
        Plots.scatter!(p2, [txs[end]], [tys[end]], markershape=:diamond, 
                       markercolor=:yellow, markersize=6, label="Final Best")
        Plots.plot!(p2, title="Trajectory: $(algo_label) on $(landscape.name)", titlefontsize=11,
                    xlims=xlims_zoom, ylims=ylims_zoom,
                    xticks=xticks_zoom, yticks=yticks_zoom,
                    aspect_ratio=:auto,
                    legend=(0.9, 1.12), legendfontsize=7,
                    top_margin=14Plots.mm, bottom_margin=12Plots.mm,
                    left_margin=15Plots.mm, right_margin=15Plots.mm)
    elseif n <= 20
        txs = [split_bits(ind, n)[1] for ind in best_trajectory]
        tys = [split_bits(ind, n)[2] for ind in best_trajectory]
        xlims_zoom, ylims_zoom = trajectory_zoom_limits(txs, tys, n)
        xticks_zoom = adaptive_axis_ticks(xlims_zoom)
        yticks_zoom = adaptive_axis_ticks(ylims_zoom)

        p2 = Plots.plot(
            txs,
            tys,
            color=:cyan,
            linewidth=0.9,
            linealpha=0.8,
            title="Blind Trajectory: $(algo_label) on $(landscape.name)",
            titlefontsize=11,
            xlabel="Left bits",
            ylabel="Right bits",
            label="Best-so-far path",
            xlims=xlims_zoom,
            ylims=ylims_zoom,
            xticks=xticks_zoom,
            yticks=yticks_zoom,
            aspect_ratio=:auto,
            legend=(0.9, 1.12),
            legendfontsize=7,
            top_margin=14Plots.mm, bottom_margin=12Plots.mm,
            left_margin=15Plots.mm, right_margin=15Plots.mm
        )
        add_trajectory_points!(p2, txs, tys)
        add_tiny_trajectory_arrows!(p2, txs, tys)
        Plots.scatter!(p2, [txs[end]], [tys[end]], markershape=:diamond,
                       markercolor=:yellow, markersize=6, label="Final Best")
    else
        if landscape.is_synthetic && landscape.n == 31
            p2 = Plots.plot(
                best_fitness_over_time,
                title="Best Fitness: $(algo_label) on $(landscape.name)",
                titlefontsize=11,
                xlabel="Generation", ylabel="Fitness", label="Best Fitness",
                color=:steelblue,
                linewidth=1.8,
                marker=:circle,
                markersize=2.2,
                ylims=(4.95, 6.05),
                yticks=5.0:0.2:6.0,
                legend=(0.17, 1.12), legendfontsize=6,
                background_color_legend=RGBA(1,1,1,0.72),
                top_margin=18Plots.mm, bottom_margin=12Plots.mm,
                left_margin=15Plots.mm, right_margin=18Plots.mm
            )
            Plots.hline!(p2, [6.0], color=:red, linestyle=:dash, label="Goal (6.0)")
            p2_right = Plots.twinx(p2)
            Plots.plot!(
                p2_right,
                Float64.(best_nones_over_time),
                color=:darkorange,
                linestyle=:dash,
                linewidth=1.2,
                label="#Active Bits",
                ylabel="Active Bits",
                ylims=(0.0, Float64(n)),
                yticks=0:5:n,
                legend=(0.83, 1.12),
                legendfontsize=6,
                background_color_legend=RGBA(1,1,1,0.72)
            )
        else
            p2 = Plots.plot(
                best_fitness_over_time,
                title="Best Fitness: $(algo_label) on $(landscape.name)",
                titlefontsize=11,
                xlabel="Generation", ylabel="Value", label="Best Fitness",
                legend=(0.9, 1.12), legendfontsize=7,
                top_margin=14Plots.mm, bottom_margin=12Plots.mm,
                left_margin=15Plots.mm, right_margin=15Plots.mm
            )
            Plots.plot!(p2, Float64.(best_nones_over_time), label="#Active Bits")
        end
    end

    # Panel 3: active-bit-count distribution over time.
    bit_count_matrix = active_bit_distribution_over_time(population_history, n)
    bit_tick_step = max(1, cld(n, 8))
    p3 = Plots.heatmap(
        1:size(bit_count_matrix, 2),
        0:n,
        bit_count_matrix,
        title="Active-Bit Distribution: $(algo_label) on $(landscape.name)",
        titlefontsize=11,
        xlabel="Generation", ylabel="Active Bits",
        colorbar_title="Individuals",
        c=:viridis,
        yticks=0:bit_tick_step:n,
        top_margin=14Plots.mm, bottom_margin=15Plots.mm,
        left_margin=15Plots.mm, right_margin=15Plots.mm
    )

    p = Plots.plot(
        p1,
        p2,
        p3,
        layout=(3, 1),
        size=(1100, 1900),
        plot_title="Dataset: $(landscape.name)",
        plot_titlefontsize=16,
        top_margin=8Plots.mm
    )

    if save_path != ""
        Plots.savefig(p, save_path)
    end

    return p
end
