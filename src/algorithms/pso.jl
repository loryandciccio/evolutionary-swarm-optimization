using Random, Statistics

"""
    run_pso(landscape::Landscape, epsilon::Float64; 
            pop_size=100, n_gens=100, w=0.7, c1=1.5, c2=1.5,
            adaptive=true, w_min=0.4, c1_end=0.5, c2_end=2.5)

Runs a Binary Particle Swarm Optimization (BPSO).
Returns a NamedTuple with best solution, histories, and parameter-control trace.
"""
function run_pso(landscape::Landscape, epsilon::Float64; 
                pop_size=100, n_gens=100, w=0.7, c1=1.5, c2=1.5,
                adaptive=true, w_min=0.4, c1_end=0.5, c2_end=2.5)
    if pop_size < 1
        error("Invalid PSO parameter pop_size=$pop_size: expected >= 1")
    end
    if n_gens < 1
        error("Invalid PSO parameter n_gens=$n_gens: expected >= 1")
    end
    for (name, value) in (("w", w), ("c1", c1), ("c2", c2), ("w_min", w_min), ("c1_end", c1_end), ("c2_end", c2_end))
        if !isfinite(value)
            error("Invalid PSO parameter $name=$value: expected a finite real number")
        end
        if value < 0
            error("Invalid PSO parameter $name=$value: expected >= 0")
        end
    end

    n = landscape.n
    if n < 1
        error("Landscape dimension n=$n is invalid for PSO: expected >= 1")
    end
    max_dec = BigInt(1) << n
    max_int = typemax(Int)
    if max_dec - 1 > max_int
        error("Landscape dimension n=$n is too large for Int-based PSO representation")
    end
    max_index = Int(max_dec - 1)
    
    # Initialize particle positions as bit vectors.
    positions = [rand(0:1, n) for _ in 1:pop_size]
    if !landscape.is_synthetic
        # Feature-selection lookup tables are defined on decimal domain 1..2^n-1;
        # the zero state is invalid.
        for i in 1:pop_size
            if bitstring_to_dec(positions[i]) == 0
                dec = rand(1:max_index)
                positions[i] = dec_to_bitstring(dec, n)
            end
        end
    end
    # Initialize particle velocities.
    velocities = [randn(n) for _ in 1:pop_size]
    
    # Deep-copy each particle so personal bests do not alias current positions.
    pbests = [copy(p) for p in positions]
    pbest_fitnesses = [evaluate_fitness(landscape, bitstring_to_dec(p), epsilon) for p in positions]
    
    gbest = copy(pbests[argmax(pbest_fitnesses)])
    gbest_fitness = maximum(pbest_fitnesses)
    
    history = Vector{Float64}[]
    pop_history = Vector{Int}[]
    w_history = Float64[]
    c1_history = Float64[]
    c2_history = Float64[]
    
    for gen in 1:n_gens
        alpha = n_gens <= 1 ? 1.0 : (gen - 1) / (n_gens - 1)
        current_w = adaptive ? (w + (w_min - w) * alpha) : w
        current_c1 = adaptive ? (c1 + (c1_end - c1) * alpha) : c1
        current_c2 = adaptive ? (c2 + (c2_end - c2) * alpha) : c2
        push!(w_history, current_w)
        push!(c1_history, current_c1)
        push!(c2_history, current_c2)

        fitnesses = Float64[]
        current_pop = Int[]
        
        for i in 1:pop_size
            # Update the velocity vector.
            r1, r2 = rand(n), rand(n)
            velocities[i] = current_w .* velocities[i] .+ 
                            current_c1 .* r1 .* (pbests[i] .- positions[i]) .+ 
                            current_c2 .* r2 .* (gbest .- positions[i])
            
            # Update the position using the Binary PSO rule.
            for d in 1:n
                sig = 1.0 / (1.0 + exp(-velocities[i][d]))
                if rand() < sig
                    positions[i][d] = 1
                else
                    positions[i][d] = 0
                end
            end
            
            # Evaluate the updated particle.
            dec = bitstring_to_dec(positions[i])
            if !landscape.is_synthetic && dec == 0
                dec = rand(1:max_index)
                positions[i] = dec_to_bitstring(dec, n)
            end
            
            fit = evaluate_fitness(landscape, dec, epsilon)
            push!(fitnesses, fit)
            push!(current_pop, dec)
            
            # Update the personal best.
            if fit > pbest_fitnesses[i]
                pbest_fitnesses[i] = fit
                pbests[i] = copy(positions[i])
            end
        end
        
        # Update the global best.
        if maximum(pbest_fitnesses) > gbest_fitness
            gbest_fitness = maximum(pbest_fitnesses)
            gbest = copy(pbests[argmax(pbest_fitnesses)])
        end
        
        push!(history, fitnesses)
        push!(pop_history, current_pop)
    end
    
    return (
        kind = :single_obj_result,
        best_ind = bitstring_to_dec(gbest),
        best_fitness = gbest_fitness,
        history = history,
        pop_history = pop_history,
        param_control_enabled = adaptive,
        param_control_history = (w = w_history, c1 = c1_history, c2 = c2_history)
    )
end

function sigmoid(x)
    return 1.0 / (1.0 + exp(-x))
end
