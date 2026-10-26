using Statistics, DataFrames, CSV, Serialization

function params_to_string(params)
    if params isa NamedTuple
        pairs_str = ["$(k)=$(v)" for (k, v) in pairs(params)]
        return isempty(pairs_str) ? "" : join(pairs_str, ";")
    end
    return string(params)
end

"""
    run_repeated_experiments(landscape::Landscape, algo_fn, n_runs=10; epsilon=0.05, kwargs...)

Runs an algorithm multiple times and collects statistics.
"""
function run_repeated_experiments(landscape::Landscape, algo_fn, n_runs=10; epsilon=0.05, kwargs...)
    if !(n_runs isa Integer)
        error("Invalid n_runs=$n_runs: expected an integer >= 1")
    end
    if n_runs < 1
        error("Invalid n_runs=$n_runs: expected >= 1")
    end

    best_fitnesses = Float64[]
    best_inds = Int[]
    histories = []
    mo_histories = []
    pop_histories = []
    param_control_histories = []
    
    for i in 1:n_runs
        println("  Run $i / $n_runs...")
        res = algo_fn(landscape; kwargs...)
        # Handle the explicit NSGA-II result contract.
        if res isa NamedTuple && get(res, :kind, nothing) == :nsga2_result
            representative = res.representative
            history = res.history
            mo_history = res.mo_history
            pop_history = res.pop_history
            best_p_ind = representative.ind
            best_p_fit = representative.scalar_fitness
            
            push!(best_inds, best_p_ind)
            push!(best_fitnesses, best_p_fit)
            push!(histories, history)
            push!(mo_histories, mo_history)
            push!(pop_histories, pop_history)
            push!(param_control_histories, get(res, :param_control_history, nothing))
        elseif res isa NamedTuple && get(res, :kind, nothing) == :single_obj_result
            best_ind = res.best_ind
            best_fit = res.best_fitness
            history = res.history
            pop_history = res.pop_history

            push!(best_inds, best_ind)
            push!(best_fitnesses, best_fit)
            push!(histories, history)
            push!(mo_histories, nothing)
            push!(pop_histories, pop_history)
            push!(param_control_histories, get(res, :param_control_history, nothing))
        else
            # Handle the legacy tuple-style result returned by older GA/PSO code paths.
            best_ind, best_fit, history, pop_history = res
            push!(best_inds, best_ind)
            push!(best_fitnesses, best_fit)
            push!(histories, history)
            push!(mo_histories, nothing)
            push!(pop_histories, pop_history)
            push!(param_control_histories, nothing)
        end
    end

    max_best = maximum(best_fitnesses)
    distinct_best_inds = sort(unique(best_inds))
    max_best_hit_count = count(x -> isapprox(x, max_best; atol=1e-12, rtol=1e-10), best_fitnesses)
    
    return (
        mean_best = mean(best_fitnesses),
        std_best = std(best_fitnesses),
        max_best = max_best,
        best_of_all = best_inds[argmax(best_fitnesses)],
        best_fitness_of_all = max_best,
        all_best_inds = best_inds,
        n_distinct_best = length(distinct_best_inds),
        distinct_best_inds = distinct_best_inds,
        max_best_hit_count = max_best_hit_count,
        histories = histories,
        mo_histories = mo_histories,
        pop_histories = pop_histories,
        param_control_histories = param_control_histories
    )
end

"""
    feature_subset_from_dec(dec::Int, n::Int)

Returns 1-based feature indices that are active in the bitstring.
"""
function feature_subset_from_dec(dec::Int, n::Int)
    return [i for i in 1:n if ((dec >> (i-1)) & 1) == 1]
end

"""
    save_results(results, name, algo_name)
"""
function save_results(results, name, algo_name; seed=nothing, seed_policy::String="unseeded-global-rng")
    mkpath("outputs/metrics")
    mkpath("outputs/runs")
    seed_str = seed === nothing ? "" : string(seed)
    
    df = DataFrame(
        landscape_name = name,
        algorithm_name = algo_name,
        scope = "train",
        seed = seed_str,
        seed_policy = seed_policy,
        mean_best = results.mean_best,
        std_best = results.std_best,
        max_best = results.max_best,
        best_ind = results.best_of_all,
        best_fitness = results.best_fitness_of_all
    )
    
    CSV.write("outputs/metrics/$(name)_$(algo_name).csv", df)
    
    # Save the full histories for later behavior plotting.
    serialize("outputs/runs/$(name)_$(algo_name).jls", results)
end

"""
    save_results(results, landscape::Landscape, algo_name)

Saves metrics including the best solution in decimal and bitstring form,
along with provenance and experiment metadata.
"""
function save_results(
    results,
    landscape::Landscape,
    algo_name;
    scope::String="train",
    params_source::String="cli",
    params_file::String="",
    manifest_path::String="",
    algo_params::NamedTuple=NamedTuple(),
    parameter_control_policy::String="none",
    seed=nothing,
    seed_policy::String="unseeded-global-rng"
)
    mkpath("outputs/metrics")
    mkpath("outputs/runs")

    scope_norm = lowercase(strip(scope))
    if !(scope_norm in ("train", "unseen"))
        error("Invalid scope '$scope'. Allowed values for metrics metadata: train, unseen")
    end

    best_ind = results.best_of_all
    n = landscape.n
    bitstring = reporting_bitstring(best_ind, n)
    distinct_best_inds = get(results, :distinct_best_inds, Int[])
    distinct_best_str = isempty(distinct_best_inds) ? "[]" : "[" * join(distinct_best_inds, ",") * "]"
    seed_str = seed === nothing ? "" : string(seed)
    hyperparams_str = params_to_string(algo_params)
    control_policy = strip(parameter_control_policy)
    control_enabled = !isempty(control_policy) && lowercase(control_policy) != "none"
    n_distinct_best = get(results, :n_distinct_best, length(distinct_best_inds))
    max_best_hit_count = get(results, :max_best_hit_count, 0)

    df = DataFrame(
        landscape_name = landscape.name,
        algorithm_name = algo_name,
        scope = scope_norm,
        params_source = params_source,
        params_file = params_file,
        manifest_path = manifest_path,
        manifest_id = isempty(manifest_path) ? "" : basename(manifest_path),
        seed = seed_str,
        seed_policy = seed_policy,
        hyperparameters = hyperparams_str,
        parameter_control_policy = control_policy,
        parameter_control_enabled = control_enabled,
        n_distinct_best = n_distinct_best,
        can_find_multiple_optima = n_distinct_best > 1,
        distinct_best_inds = distinct_best_str,
        max_best_hit_count = max_best_hit_count,
        mean_best = results.mean_best,
        std_best = results.std_best,
        max_best = results.max_best,
        best_ind = best_ind,
        best_bitstring = bitstring,
        best_fitness = results.best_fitness_of_all
    )

    CSV.write("outputs/metrics/$(landscape.name)_$(algo_name).csv", df)
    serialize("outputs/runs/$(landscape.name)_$(algo_name).jls", results)
end
