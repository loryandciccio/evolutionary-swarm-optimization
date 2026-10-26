using Random, Statistics

"""
    run_ga(landscape::Landscape, epsilon::Float64; 
           pop_size=100, n_gens=100, p_mut=0.01, p_cross=0.8, 
           tournament_size=3, adaptive=true)

Runs a standard Genetic Algorithm for feature selection.
Returns a NamedTuple with best solution, histories, and parameter-control trace.
"""
function run_ga(landscape::Landscape, epsilon::Float64; 
                pop_size=100, n_gens=100, p_mut=0.01, p_cross=0.8, 
                tournament_size=3, adaptive=true)
    if pop_size < 1
        error("Invalid GA parameter pop_size=$pop_size: expected >= 1")
    end
    if n_gens < 1
        error("Invalid GA parameter n_gens=$n_gens: expected >= 1")
    end
    if tournament_size < 1
        error("Invalid GA parameter tournament_size=$tournament_size: expected >= 1")
    end
    if !(0.0 <= p_mut <= 1.0)
        error("Invalid GA parameter p_mut=$p_mut: expected in [0, 1]")
    end
    if !(0.0 <= p_cross <= 1.0)
        error("Invalid GA parameter p_cross=$p_cross: expected in [0, 1]")
    end

    n = landscape.n
    if n < 1
        error("Landscape dimension n=$n is invalid for GA: expected >= 1")
    end
    max_dec = BigInt(1) << n
    max_int = typemax(Int)
    if max_dec - 1 > max_int
        error("Landscape dimension n=$n is too large for Int-based GA representation")
    end
    max_index = Int(max_dec - 1)
    
    # Initialize the population in decimal representation.
    start_dec = landscape.is_synthetic ? 0 : 1
    pop = rand(start_dec:max_index, pop_size)
    
    history = Vector{Float64}[]
    pop_history = Vector{Int}[]
    p_mut_history = Float64[]
    
    best_ind = -1
    best_fitness = -Inf
    
    for gen in 1:n_gens
        # Evaluate the current population.
        fitnesses = [evaluate_fitness(landscape, ind, epsilon) for ind in pop]
        
        # Record per-generation statistics.
        push!(history, fitnesses)
        push!(pop_history, copy(pop))
        
        # Update the overall best solution.
        current_best_idx = argmax(fitnesses)
        if fitnesses[current_best_idx] > best_fitness
            best_fitness = fitnesses[current_best_idx]
            best_ind = pop[current_best_idx]
        end
        
        # Increase mutation when fitness diversity becomes too small.
        current_p_mut = p_mut
        if adaptive
            fit_std = std(fitnesses)
            if fit_std < 0.001
                current_p_mut = min(0.5, p_mut * 5.0)
            end
        end
        push!(p_mut_history, current_p_mut)

        if gen == n_gens
            continue
        end
        
        # Selection and reproduction.
        new_pop = Int[]
        
        # Elitism: preserve the best individual.
        push!(new_pop, pop[current_best_idx])
        
        while length(new_pop) < pop_size
            # Tournament selection.
            p1 = tournament_selection(pop, fitnesses, tournament_size)
            p2 = tournament_selection(pop, fitnesses, tournament_size)
            
            # One-point crossover.
            c1, c2 = p1, p2
            if rand() < p_cross
                c1, c2 = crossover_one_point(p1, p2, n)
            end
            
            # Bit-flip mutation.
            c1 = mutate_bitflip(c1, n, current_p_mut)
            c2 = mutate_bitflip(c2, n, current_p_mut)
            
            # Prevent the invalid zero state in feature-selection landscapes.
            if !landscape.is_synthetic
                if c1 == 0 c1 = rand(1:max_index) end
                if c2 == 0 c2 = rand(1:max_index) end
            end
            
            push!(new_pop, c1)
            if length(new_pop) < pop_size
                push!(new_pop, c2)
            end
        end
        pop = new_pop
    end
    
    return (
        kind = :single_obj_result,
        best_ind = best_ind,
        best_fitness = best_fitness,
        history = history,
        pop_history = pop_history,
        param_control_enabled = adaptive,
        param_control_history = (p_mut = p_mut_history,)
    )
end

function tournament_selection(pop, fitnesses, k)
    indices = rand(1:length(pop), k)
    best_idx = indices[1]
    best_f = fitnesses[best_idx]
    for i in 2:k
        if fitnesses[indices[i]] > best_f
            best_idx = indices[i]
            best_f = fitnesses[best_idx]
        end
    end
    return pop[best_idx]
end

function crossover_one_point(p1, p2, n)
    if n <= 1
        return p1, p2
    end
    point = rand(1:(n-1))
    mask_low = (1 << point) - 1
    mask_high = xor(((1 << n) - 1), mask_low)
    
    c1 = (p1 & mask_low) | (p2 & mask_high)
    c2 = (p2 & mask_low) | (p1 & mask_high)
    
    return c1, c2
end

function mutate_bitflip(ind, n, p_mut)
    for i in 1:n
        if rand() < p_mut
            ind = xor(ind, (1 << (i-1)))
        end
    end
    return ind
end
