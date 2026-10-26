"""
    evaluate_fitness(landscape::Landscape, x_dec::Int, epsilon::Float64)

Evaluates the fitness of a combination represented by its decimal index.
For FS landscapes, index 1 corresponds to dec 1.
For synthetic, index 1 corresponds to dec 0.
"""
function validate_dec_domain(landscape::Landscape, x_dec::Int)
    if landscape.n < 1
        error("Invalid landscape dimension n=$(landscape.n) for landscape '$(landscape.name)': expected >= 1")
    end

    max_dec = (BigInt(1) << landscape.n) - 1
    x_big = BigInt(x_dec)
    if landscape.is_synthetic
        if x_big < 0 || x_big > max_dec
            error("Invalid synthetic state x_dec=$x_dec for landscape '$(landscape.name)': valid domain is 0..2^n-1")
        end
    else
        if x_big < 1 || x_big > max_dec
            error("Invalid FS state x_dec=$x_dec for landscape '$(landscape.name)': valid domain is 1..2^n-1")
        end
    end
end

function evaluate_fitness(landscape::Landscape, x_dec::Int, epsilon::Float64)
    validate_dec_domain(landscape, x_dec)
    if landscape.is_synthetic
        if isempty(landscape.lookup_table)
            val = triangle_value(count_ones(x_dec), landscape.triangle_m, landscape.triangle_s)
        else
            val = landscape.lookup_table[x_dec + 1]
        end
    else
        val = landscape.lookup_table[x_dec]
    end
    
    # Penalty term: fraction of active bits.
    n_features = count_ones(x_dec)
    penalty = n_features / landscape.n
    
    return val - epsilon * penalty
end

"""
    evaluate_multiobjective(landscape::Landscape, x_dec::Int; penalty_weight::Float64=1.0)

Returns two objectives for NSGA-II minimization:
1. Error = 1 - Accuracy
2. Weighted penalty = penalty_weight * (n_features / n)
"""
function evaluate_multiobjective(landscape::Landscape, x_dec::Int; penalty_weight::Float64=1.0)
    validate_dec_domain(landscape, x_dec)
    if landscape.is_synthetic
        if isempty(landscape.lookup_table)
            acc = triangle_value(count_ones(x_dec), landscape.triangle_m, landscape.triangle_s)
        else
            acc = landscape.lookup_table[x_dec + 1]
        end
    else
        acc = landscape.lookup_table[x_dec]
    end
    
    obj1 = 1.0 - acc
    obj2 = penalty_weight * (Float64(count_ones(x_dec)) / landscape.n)
    
    return (obj1, obj2) # Both objectives are minimized.
end

"""
    bitstring_to_dec(bits::AbstractVector{<:Integer})
"""
function bitstring_to_dec(bits::AbstractVector{<:Integer})
    if isempty(bits)
        error("Invalid bitstring: expected at least one bit")
    end
    dec_big = BigInt(0)
    for (i, b) in enumerate(bits)
        if b != 0 && b != 1
            error("Invalid bitstring value at index $i: expected 0 or 1, got $b")
        end
        if b == 1
            dec_big += (BigInt(1) << (i - 1))
        end
    end
    if dec_big > typemax(Int)
        error("Bitstring dimension $(length(bits)) is too large for Int representation")
    end
    return Int(dec_big)
end

"""
    dec_to_bitstring(dec::Int, n::Int)
"""
function dec_to_bitstring(dec::Int, n::Int)
    if n < 1
        error("Invalid bitstring length n=$n: expected >= 1")
    end
    max_dec = (BigInt(1) << n) - 1
    dec_big = BigInt(dec)
    if dec_big < 0 || dec_big > max_dec
        error("Invalid decimal state dec=$dec for n=$n: valid domain is 0..2^n-1")
    end

    bits = zeros(Int, n)
    for i in 1:n
        if (dec >> (i-1)) & 1 == 1
            bits[i] = 1
        end
    end
    return bits
end

"""
    reporting_bitstring(dec::Int, n::Int)

Returns the bitstring in standard human-readable form (most significant bit on the left).
Internal search operators still use the little-endian vector returned by `dec_to_bitstring`.
"""
function reporting_bitstring(dec::Int, n::Int)
    return join(reverse(dec_to_bitstring(dec, n)), "")
end
