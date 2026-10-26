"""
    detect_local_optima(landscape::Landscape, epsilon::Float64)

Finds all local optima in the landscape.
A point x is a local optimum if f(x) >= f(y) for all neighbors y in Hamming distance 1.
"""
function detect_local_optima(landscape::Landscape, epsilon::Float64)
    n = landscape.n
    if n > 22
        error("Exact local-optima detection is disabled for n > 22 to avoid exhaustive search on huge landscapes")
    end
    num_combinations = 2^n
    local_optima = Int[]
    
    # Work directly on decimal state indices:
    # Feature-selection landscapes use 1 to 2^n - 1.
    # Synthetic landscapes use 0 to 2^n - 1.
    start_dec = landscape.is_synthetic ? 0 : 1
    end_dec = num_combinations - 1
    
    # Precompute fitness values to avoid repeated evaluations.
    fitnesses = zeros(Float64, num_combinations)
    Threads.@threads for i in start_dec:end_dec
        fitnesses[i+1] = evaluate_fitness(landscape, i, epsilon)
    end
    
    # Boolean flags used to mark whether each state remains a local optimum.
    is_optimum = fill(true, num_combinations)
    # The zero state is invalid for feature-selection lookup tables.
    if !landscape.is_synthetic
        is_optimum[1] = false
    end
    
    Threads.@threads for idx in start_dec:end_dec
        f_idx = fitnesses[idx+1]
        
        # Check every Hamming-1 neighbor.
        for bit_idx in 1:n
            # Flip one bit to generate a neighbor.
            mask = 1 << (bit_idx-1)
            neighbor_idx = xor(idx, mask)
            
            # Skip the invalid zero state in feature-selection landscapes.
            if !landscape.is_synthetic && neighbor_idx == 0
                continue
            end
            
            # A strictly better neighbor disqualifies the current state.
            if fitnesses[neighbor_idx+1] > f_idx
                is_optimum[idx+1] = false
                break
            end
        end
    end
    
    # Collect the surviving local optima.
    for i in start_dec:end_dec
        if is_optimum[i+1]
            push!(local_optima, i)
        end
    end
    
    return local_optima
end
