using HDF5, Statistics

"""
    Landscape

A struct to represent an optimization landscape.
- `n`: Number of bits (features).
- `lookup_table`: Vector of mean accuracies (size 2^n - 1 for FS, 2^n for synthetic).
- `name`: Name of the dataset.
- `is_synthetic`: Boolean indicating if it's the triangle function.
- `triangle_m`: Triangle-wave slope parameter (synthetic only).
- `triangle_s`: Triangle-wave period-to-reversal parameter (synthetic only).
"""
struct Landscape{T<:Number}
    n::Int
    lookup_table::Vector{T}
    name::String
    is_synthetic::Bool
    triangle_m::Float64
    triangle_s::Int
end

"""
    load_h5_landscape(file_path::String, name::String)

Loads an HDF5 file and computes the mean of the accuracies.
Returns a Landscape{Float64}.
"""
function load_h5_landscape(file_path::String, name::String)
    data = h5open(file_path, "r") do file
        if !haskey(file, "accuracies")
            error("Missing 'accuracies' dataset in HDF5 file: $file_path")
        end
        # The dataset layout is (N, runs).
        acc = read(file["accuracies"])
        if ndims(acc) != 2
            error("'accuracies' must be a 2D dataset in HDF5 file: $file_path")
        end
        # Compute the mean accuracy for each row.
        vec_mean = dropdims(mean(acc, dims=2), dims=2)
        return vec_mean
    end
    
    # Infer n from the lookup-table size:
    # if size = 2^n - 1, then size + 1 = 2^n and n = log2(size + 1).
    len = length(data)
    if len < 1
        error("Invalid FS lookup-table length ($len) in $file_path: expected 2^n - 1 with n >= 1")
    end
    if !ispow2(len + 1)
        error("Invalid FS lookup-table length ($len) in $file_path: expected 2^n - 1")
    end
    n = trailing_zeros(len + 1)
    
    return Landscape{Float64}(n, data, name, false, NaN, 0)
end

"""
    triangle_value(n_ones::Int, m::Float64, s::Int)

Computes the synthetic triangle value as a function of the number of active bits.
"""
function triangle_value(n_ones::Int, m::Float64, s::Int)
    if s <= 0
        error("Parameter s must be > 0 for the triangle landscape")
    end
    if n_ones == 0
        return 0.0
    end

    idx_ceil = cld(n_ones, s)
    if idx_ceil % 2 == 1
        if n_ones % s == 0
            return m * s
        end
        return m * (n_ones % s)
    end
    return m * (idx_ceil * s - n_ones)
end

"""
    create_triangle_landscape(n::Int, m::Float64, s::Int)

Generates the triangle function landscape.
"""
function create_triangle_landscape(n::Int, m::Float64, s::Int; name::String="Triangle", precompute::Bool=true)
    if n < 1
        error("n must be >= 1 for the triangle landscape")
    end
    if s <= 0
        error("Parameter s must be > 0 for the triangle landscape")
    end

    lookup_table = Float64[]
    if precompute
        num_combinations = 2^n
        lookup_table = zeros(Float64, num_combinations)
        for i in 0:(num_combinations-1)
            lookup_table[i+1] = triangle_value(count_ones(i), m, s)
        end
    end
    
    return Landscape{Float64}(n, lookup_table, name, true, m, s)
end

"""
    create_asymmetric_triangle_landscape(n::Int)

Generates the specific asymmetric triangle function for Step 6 (n=31).
Uses UInt8 to save memory (2 GiB for n=31).
"""
function create_asymmetric_triangle_landscape(n::Int; name::String="Unreleased-Test-Landscape")
    error("The course-provided hidden-test landscape definition is excluded from the public edition. Use the synthetic demo instead.")
end
