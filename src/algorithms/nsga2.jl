using Random, Statistics

scalarize_project_objectives(obj::Tuple{Float64, Float64}) = 1.0 - obj[1] - obj[2]

"""
    run_nsga2(landscape::Landscape; 
              pop_size=100, n_gens=100, p_mut=0.01, p_cross=0.8, epsilon=0.05, adaptive=true)

Runs NSGA-II for multi-objective optimization.
Objectives (to minimize):
1. Error (1 - Accuracy)
2. Weighted penalty objective used by the project fitness convention
Returns:
- pareto_front: final non-dominated solutions
- representative: deterministic representative from the final Pareto front
  chosen by maximum scalarized fitness (accuracy - ε*penalty)
- history: scalarized per-generation history (kept for compatibility)
- mo_history: per-generation objective pairs (error, weighted penalty)
- pop_history: per-generation populations
- param_control_history: per-generation adaptive values and MO trigger traces
"""
function run_nsga2(landscape::Landscape; 
                   pop_size=100, n_gens=100, p_mut=0.01, p_cross=0.8, epsilon=0.05, adaptive=true)
    if pop_size < 1
        error("Invalid NSGA-II parameter pop_size=$pop_size: expected >= 1")
    end
    if n_gens < 1
        error("Invalid NSGA-II parameter n_gens=$n_gens: expected >= 1")
    end
    if !(0.0 <= p_mut <= 1.0)
        error("Invalid NSGA-II parameter p_mut=$p_mut: expected in [0, 1]")
    end
    if !(0.0 <= p_cross <= 1.0)
        error("Invalid NSGA-II parameter p_cross=$p_cross: expected in [0, 1]")
    end

    n = landscape.n
    if n < 1
        error("Landscape dimension n=$n is invalid for NSGA-II: expected >= 1")
    end
    max_dec = BigInt(1) << n
    max_int = typemax(Int)
    if max_dec - 1 > max_int
        error("Landscape dimension n=$n is too large for Int-based NSGA-II representation")
    end
    max_index = Int(max_dec - 1)
    start_dec = landscape.is_synthetic ? 0 : 1
    
    # Initialize the population.
    pop = rand(start_dec:max_index, pop_size)
    
    history = Vector{Float64}[] # Scalarized fitness history for tracking and plotting.
    mo_history = Vector{Tuple{Float64, Float64}}[]
    pop_history = Vector{Int}[]
    p_mut_history = Float64[]
    p_cross_history = Float64[]
    front_diversity_history = Float64[]
    front_size_history = Int[]

    mo_diversity_threshold = 0.02
    mo_min_front_size = 3
    
    for gen in 1:n_gens
        # Evaluate the current population.
        objs = [evaluate_multiobjective(landscape, ind; penalty_weight=epsilon) for ind in pop]
        ranks, distances = compute_nsga2_stats(objs)
        
        # Track scalarized fitness under the same convention as evaluate_fitness.
        scalar_f = [scalarize_project_objectives(o) for o in objs]
        push!(history, scalar_f)
        push!(mo_history, copy(objs))
        push!(pop_history, copy(pop))

        # Use the current non-dominated front geometry as the adaptive trigger.
        front_diversity, front_size = mo_front_diversity(objs, ranks)
        push!(front_diversity_history, front_diversity)
        push!(front_size_history, front_size)

        current_p_mut = p_mut
        current_p_cross = p_cross
        if adaptive
            if front_size < mo_min_front_size || front_diversity < mo_diversity_threshold
                current_p_mut = min(0.5, p_mut * 5.0)
                current_p_cross = max(0.5, p_cross * 0.8)
            end
        end
        push!(p_mut_history, current_p_mut)
        push!(p_cross_history, current_p_cross)

        if gen == n_gens
            continue
        end
        
        # Generate offspring.
        offspring = Int[]
        while length(offspring) < pop_size
            # Tournament selection based on rank and crowding distance.
            p1 = tournament_selection_nsga2(pop, ranks, distances, 3)
            p2 = tournament_selection_nsga2(pop, ranks, distances, 3)
            
            c1, c2 = p1, p2
            if rand() < current_p_cross
                c1, c2 = crossover_one_point(p1, p2, n)
            end
            c1 = mutate_bitflip(c1, n, current_p_mut)
            c2 = mutate_bitflip(c2, n, current_p_mut)
            
            if !landscape.is_synthetic
                if c1 == 0 c1 = rand(1:max_index) end
                if c2 == 0 c2 = rand(1:max_index) end
            end
            
            push!(offspring, c1)
            if length(offspring) < pop_size
                push!(offspring, c2)
            end
        end
        
        # Combine parents and offspring into a population of size 2N.
        combined_pop = [pop; offspring]
        combined_objs = [objs; [evaluate_multiobjective(landscape, ind; penalty_weight=epsilon) for ind in offspring]]
        
        # Select the next population using standard NSGA-II survival.
        ranks, distances = compute_nsga2_stats(combined_objs)
        
        new_pop_indices = Int[]
        current_rank = 1
        while length(new_pop_indices) < pop_size
            rank_indices = findall(r -> r == current_rank, ranks)
            if length(new_pop_indices) + length(rank_indices) <= pop_size
                append!(new_pop_indices, rank_indices)
            else
                # Break ties in the last accepted rank by crowding distance.
                sorted_rank_indices = rank_indices[sortperm(distances[rank_indices], rev=true)]
                remaining = pop_size - length(new_pop_indices)
                append!(new_pop_indices, sorted_rank_indices[1:remaining])
            end
            current_rank += 1
        end
        
        pop = combined_pop[new_pop_indices]
    end
    
    # Extract the final Pareto front.
    final_objs = [evaluate_multiobjective(landscape, ind; penalty_weight=epsilon) for ind in pop]
    ranks, _ = compute_nsga2_stats(final_objs)
    pareto_indices = findall(r -> r == 1, ranks)
    pareto_front = pop[pareto_indices]
    pareto_objs = final_objs[pareto_indices]
    
    # Select a deterministic representative from the final Pareto front
    # using the project convention accuracy - ε*penalty.
    scalar_scores = [scalarize_project_objectives(o) for o in pareto_objs]
    rep_idx = argmax(scalar_scores)
    rep_bits = dec_to_bitstring(pareto_front[rep_idx], n)
    rep_features = [i for i in 1:n if rep_bits[i] == 1]
    representative = (
        ind = pareto_front[rep_idx],
        bitstring = rep_bits,
        features = rep_features,
        objectives = pareto_objs[rep_idx], # (error, weighted penalty)
        scalar_fitness = scalar_scores[rep_idx]
    )
    
    return (
        kind = :nsga2_result,
        pareto_front = pareto_front,
        representative = representative,
        history = history,
        mo_history = mo_history,
        pop_history = pop_history,
        param_control_enabled = adaptive,
        param_control_history = (
            p_mut = p_mut_history,
            p_cross = p_cross_history,
            mo_front_diversity = front_diversity_history,
            mo_front_size = front_size_history
        )
    )
end

function mo_front_diversity(objs, ranks)
    front_idx = findall(r -> r == 1, ranks)
    if isempty(front_idx)
        return 0.0, 0
    end
    front_objs = objs[front_idx]
    errs = [o[1] for o in front_objs]
    pens = [o[2] for o in front_objs]
    err_spread = maximum(errs) - minimum(errs)
    pen_spread = maximum(pens) - minimum(pens)
    # Use Euclidean spread over (error, weighted penalty);
    # low spread indicates a collapsed or stagnating front.
    diversity = sqrt(err_spread^2 + pen_spread^2)
    return diversity, length(front_idx)
end

function compute_nsga2_stats(objs)
    n_pop = length(objs)
    ranks = zeros(Int, n_pop)
    distances = zeros(Float64, n_pop)
    
    # 1. Non-dominated sorting.
    S = [Int[] for _ in 1:n_pop]
    n = zeros(Int, n_pop)
    fronts = [Int[]]
    
    for i in 1:n_pop
        for j in 1:n_pop
            if dominates(objs[i], objs[j])
                push!(S[i], j)
            elseif dominates(objs[j], objs[i])
                n[i] += 1
            end
        end
        if n[i] == 0
            ranks[i] = 1
            push!(fronts[1], i)
        end
    end
    
    k = 1
    while !isempty(fronts[k])
        next_front = Int[]
        for i in fronts[k]
            for j in S[i]
                n[j] -= 1
                if n[j] == 0
                    ranks[j] = k + 1
                    push!(next_front, j)
                end
            end
        end
        k += 1
        push!(fronts, next_front)
    end
    
    # 2. Crowding distance.
    for front in fronts
        if isempty(front) continue end
        len = length(front)
        if len <= 2
            distances[front] .= Inf
            continue
        end
        
        for m in 1:length(objs[1])
            # Sort by objective m.
            sorted_indices = front[sortperm([objs[i][m] for i in front])]
            distances[sorted_indices[1]] = Inf
            distances[sorted_indices[end]] = Inf
            
            f_min = objs[sorted_indices[1]][m]
            f_max = objs[sorted_indices[end]][m]
            denom = f_max - f_min
            if denom == 0 continue end
            
            for i in 2:(len-1)
                distances[sorted_indices[i]] += (objs[sorted_indices[i+1]][m] - objs[sorted_indices[i-1]][m]) / denom
            end
        end
    end
    
    return ranks, distances
end

function dominates(o1, o2)
    # Both objectives are minimized.
    all(o1 .<= o2) && any(o1 .< o2)
end

function tournament_selection_nsga2(pop, ranks, distances, k)
    indices = rand(1:length(pop), k)
    best_idx = indices[1]
    
    for i in 2:k
        curr = indices[i]
        if ranks[curr] < ranks[best_idx]
            best_idx = curr
        elseif ranks[curr] == ranks[best_idx] && distances[curr] > distances[best_idx]
            best_idx = curr
        end
    end
    return pop[best_idx]
end
