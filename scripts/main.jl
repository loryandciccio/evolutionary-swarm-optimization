include("../src/Project3.jl")
using .Project3
using Dates, TOML
using Random

mkpath("outputs/plots")
mkpath("outputs/metrics")
mkpath("outputs/tables")
mkpath("outputs/runs")
mkpath("outputs/params")
mkpath("outputs/logs")

function parse_arg(args::Vector{String}, key::String, default::String)
    for a in args
        if startswith(a, key * "=")
            return String(split(a, "=", limit=2)[2])
        end
    end
    return default
end

function arg_provided(args::Vector{String}, key::String)
    return any(startswith(a, key * "=") for a in args)
end

function parse_bool(s::String)
    t = lowercase(strip(s))
    return t in ("1", "true", "yes", "y")
end

const BUDGET_RULE_FORMULA = "B(n)=min(n^2, 400)"

function problem_budget(n::Int)
    return min(n * n, 400)
end

function effective_budget_rows(landscapes)
    rows = Vector{Dict{String, Any}}()
    for ls in landscapes
        b = problem_budget(ls.n)
        push!(rows, Dict(
            "landscape" => ls.name,
            "n" => ls.n,
            "budget" => b,
            "ga_pop_size" => b,
            "ga_n_gens" => b,
            "nsga2_pop_size" => b,
            "nsga2_n_gens" => b,
            "pso_pop_size" => b,
            "pso_n_gens" => b
        ))
    end
    return rows
end

function nt_to_dict(nt::NamedTuple)
    d = Dict{String, Any}()
    for (k, v) in pairs(nt)
        d[string(k)] = v
    end
    return d
end

function save_params_snapshot(path::String, mode::String, epsilon, ga_params, nsga2_params, pso_params, landscapes)
    mkpath(dirname(path))
    data = Dict(
        "metadata" => Dict(
            "saved_at_utc" => Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ"),
            "generated_mode" => mode
        ),
        "budget_policy" => Dict(
            "formula" => BUDGET_RULE_FORMULA,
            "notes" => "pop_size and n_gens are derived from landscape dimensionality n for GA/NSGA-II/PSO."
        ),
        "effective_landscape_budgets" => effective_budget_rows(landscapes),
        "global" => Dict("epsilon" => epsilon),
        "ga" => nt_to_dict(ga_params),
        "nsga2" => nt_to_dict(nsga2_params),
        "pso" => nt_to_dict(pso_params)
    )
    open(path, "w") do io
        TOML.print(io, data)
    end
    println("Saved parameter snapshot to $path")
end

function read_params_snapshot(path::String)
    if !isfile(path)
        error("Parameter snapshot file not found: $path")
    end
    data = TOML.parsefile(path)

    if !haskey(data, "global") || !haskey(data["global"], "epsilon")
        error("Invalid params file '$path': missing [global].epsilon")
    end
    for sec in ("ga", "nsga2", "pso")
        if !haskey(data, sec)
            error("Invalid params file '$path': missing [$sec] section")
        end
    end

    ga = data["ga"]
    nsga2 = data["nsga2"]
    pso = data["pso"]

    return (
        epsilon = Float64(data["global"]["epsilon"]),
        ga_params = (
            pop_size = Int(ga["pop_size"]),
            n_gens = Int(ga["n_gens"]),
            p_mut = Float64(ga["p_mut"]),
            p_cross = Float64(ga["p_cross"]),
            tournament_size = Int(ga["tournament_size"]),
            adaptive = haskey(ga, "adaptive") ? Bool(ga["adaptive"]) : true
        ),
        nsga2_params = (
            pop_size = Int(nsga2["pop_size"]),
            n_gens = Int(nsga2["n_gens"]),
            p_mut = Float64(nsga2["p_mut"]),
            p_cross = Float64(nsga2["p_cross"]),
            adaptive = haskey(nsga2, "adaptive") ? Bool(nsga2["adaptive"]) : true
        ),
        pso_params = (
            pop_size = Int(pso["pop_size"]),
            n_gens = Int(pso["n_gens"]),
            w = Float64(pso["w"]),
            c1 = Float64(pso["c1"]),
            c2 = Float64(pso["c2"]),
            adaptive = haskey(pso, "adaptive") ? Bool(pso["adaptive"]) : true,
            w_min = haskey(pso, "w_min") ? Float64(pso["w_min"]) : 0.4,
            c1_end = haskey(pso, "c1_end") ? Float64(pso["c1_end"]) : 0.5,
            c2_end = haskey(pso, "c2_end") ? Float64(pso["c2_end"]) : 2.5
        )
    )
end

function assert_no_snapshot_conflicts(args::Vector{String}, cli, frozen)
    conflicts = String[]
    # In Step 6, epsilon is treated as a property of the problem itself,
    # so CLI overrides remain allowed even when other hyperparameters are frozen.
    checks = [
        ("--ga-pmut", "ga.p_mut", cli.ga_params.p_mut, frozen.ga_params.p_mut),
        ("--ga-pcross", "ga.p_cross", cli.ga_params.p_cross, frozen.ga_params.p_cross),
        ("--ga-tour", "ga.tournament_size", cli.ga_params.tournament_size, frozen.ga_params.tournament_size),
        ("--ga-adaptive", "ga.adaptive", cli.ga_params.adaptive, frozen.ga_params.adaptive),
        ("--nsga-pmut", "nsga2.p_mut", cli.nsga2_params.p_mut, frozen.nsga2_params.p_mut),
        ("--nsga-pcross", "nsga2.p_cross", cli.nsga2_params.p_cross, frozen.nsga2_params.p_cross),
        ("--nsga-adaptive", "nsga2.adaptive", cli.nsga2_params.adaptive, frozen.nsga2_params.adaptive),
        ("--pso-w", "pso.w", cli.pso_params.w, frozen.pso_params.w),
        ("--pso-c1", "pso.c1", cli.pso_params.c1, frozen.pso_params.c1),
        ("--pso-c2", "pso.c2", cli.pso_params.c2, frozen.pso_params.c2),
        ("--pso-adaptive", "pso.adaptive", cli.pso_params.adaptive, frozen.pso_params.adaptive),
        ("--pso-w-min", "pso.w_min", cli.pso_params.w_min, frozen.pso_params.w_min),
        ("--pso-c1-end", "pso.c1_end", cli.pso_params.c1_end, frozen.pso_params.c1_end),
        ("--pso-c2-end", "pso.c2_end", cli.pso_params.c2_end, frozen.pso_params.c2_end)
    ]
    for (flag, name, cli_val, frozen_val) in checks
        if arg_provided(args, flag) && cli_val != frozen_val
            push!(conflicts, "$flag ($name): CLI=$cli_val, frozen=$frozen_val")
        end
    end

    if !isempty(conflicts)
        msg = join(conflicts, "\n  - ")
        error("Conflicting CLI tuning with --params-file. Remove conflicting flags or match the snapshot exactly:\n  - $msg")
    end
end

function assert_no_manual_budget_overrides(args::Vector{String})
    blocked = [
        "--ga-pop", "--ga-gens",
        "--nsga-pop", "--nsga-gens",
        "--pso-pop", "--pso-gens"
    ]
    provided = [f for f in blocked if arg_provided(args, f)]
    if !isempty(provided)
        joined = join(provided, ", ")
        error("Population/generation overrides are disabled. Use the global budget rule $BUDGET_RULE_FORMULA for all algorithms. Remove: $joined")
    end
end

function write_run_manifest(
    mode::String,
    landscapes,
    epsilon,
    ga_params,
    nsga2_params,
    pso_params;
    params_source::String,
    params_file::String,
    seed=nothing,
    seed_policy::String="unseeded-global-rng"
)
    ts = Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHHMMSSZ")
    manifest_path = "outputs/runs/run_manifest_$ts.toml"
    data = Dict(
        "metadata" => Dict(
            "saved_at_utc" => Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ"),
            "mode" => mode,
            "params_source" => params_source,
            "params_file" => params_file,
            "seed" => seed === nothing ? "" : string(seed),
            "seed_policy" => seed_policy
        ),
        "budget_policy" => Dict(
            "formula" => BUDGET_RULE_FORMULA,
            "notes" => "pop_size and n_gens are derived from landscape dimensionality n for GA/NSGA-II/PSO."
        ),
        "effective_landscape_budgets" => effective_budget_rows(landscapes),
        "landscapes" => [ls.name for ls in landscapes],
        "global" => Dict("epsilon" => epsilon),
        "ga" => nt_to_dict(ga_params),
        "nsga2" => nt_to_dict(nsga2_params),
        "pso" => nt_to_dict(pso_params)
    )
    open(manifest_path, "w") do io
        TOML.print(io, data)
    end
    open("outputs/runs/latest_run_manifest.toml", "w") do io
        TOML.print(io, data)
    end
    println("Run manifest saved to $manifest_path")
    return manifest_path
end

args = ARGS
assert_no_manual_budget_overrides(args)
mode = parse_arg(args, "--mode", "train")
epsilon = parse(Float64, parse_arg(args, "--epsilon", "0.05"))
n_runs = parse(Int, parse_arg(args, "--runs", "10"))
params_file = parse_arg(args, "--params-file", "")
save_params_file = parse_arg(args, "--save-params-file", "")
require_frozen = parse_bool(parse_arg(args, "--require-frozen", "true"))
unseen_behavior_plots = parse_bool(parse_arg(args, "--unseen-behavior-plots", "true"))
seed_str = parse_arg(args, "--seed", "")
seed = nothing
if !isempty(strip(seed_str))
    parsed_seed = tryparse(Int, seed_str)
    if parsed_seed === nothing
        error("Invalid --seed value '$seed_str'. Expected an integer.")
    end
    seed = parsed_seed
end

# Reproducibility policy:
# If --seed is provided, the global RNG is seeded once at startup and the full pipeline
# (all landscapes, algorithms, and repeated runs in fixed order) consumes a deterministic RNG stream.
seed_policy = seed === nothing ? "unseeded-global-rng" : "single-top-level-seed-global-rng-stream"
if seed !== nothing
    Random.seed!(seed)
    println("Using deterministic seed: $seed")
end

ga_params = (
    pop_size = parse(Int, parse_arg(args, "--ga-pop", "100")),
    n_gens = parse(Int, parse_arg(args, "--ga-gens", "100")),
    p_mut = parse(Float64, parse_arg(args, "--ga-pmut", "0.01")),
    p_cross = parse(Float64, parse_arg(args, "--ga-pcross", "0.8")),
    tournament_size = parse(Int, parse_arg(args, "--ga-tour", "3")),
    adaptive = parse_bool(parse_arg(args, "--ga-adaptive", "true"))
)
nsga2_params = (
    pop_size = parse(Int, parse_arg(args, "--nsga-pop", "100")),
    n_gens = parse(Int, parse_arg(args, "--nsga-gens", "100")),
    p_mut = parse(Float64, parse_arg(args, "--nsga-pmut", "0.01")),
    p_cross = parse(Float64, parse_arg(args, "--nsga-pcross", "0.8")),
    adaptive = parse_bool(parse_arg(args, "--nsga-adaptive", "true"))
)
pso_params = (
    pop_size = parse(Int, parse_arg(args, "--pso-pop", "100")),
    n_gens = parse(Int, parse_arg(args, "--pso-gens", "100")),
    w = parse(Float64, parse_arg(args, "--pso-w", "0.7")),
    c1 = parse(Float64, parse_arg(args, "--pso-c1", "1.5")),
    c2 = parse(Float64, parse_arg(args, "--pso-c2", "1.5")),
    adaptive = parse_bool(parse_arg(args, "--pso-adaptive", "true")),
    w_min = parse(Float64, parse_arg(args, "--pso-w-min", "0.4")),
    c1_end = parse(Float64, parse_arg(args, "--pso-c1-end", "0.5")),
    c2_end = parse(Float64, parse_arg(args, "--pso-c2-end", "2.5"))
)

params_source = "cli"
if mode == "unseen"
    if require_frozen && isempty(params_file)
        error("Unseen mode requires frozen parameters by default. Provide --params-file=<path> or set --require-frozen=false explicitly.")
    end
    if !isempty(params_file)
        cli_values = (epsilon=epsilon, ga_params=ga_params, nsga2_params=nsga2_params, pso_params=pso_params)
        frozen = read_params_snapshot(params_file)
        assert_no_snapshot_conflicts(args, cli_values, frozen)
        # Even if the frozen snapshot used epsilon > 0, Step 6 requires epsilon = 0
        # for both hidden feature-selection landscapes and the hidden synthetic landscape.
        ga_params = frozen.ga_params
        nsga2_params = frozen.nsga2_params
        pso_params = frozen.pso_params
        params_source = "params-file"
        println("Using frozen parameters from $params_file")
    else
        println("WARNING: running unseen mode without frozen params (--require-frozen=false).")
    end
    println("FORCED epsilon=0.0 for unseen mode (Step 6 requirements).")
    epsilon = 0.0
end

function load_train_landscapes()
    return [
        Project3.load_h5_landscape("train data/01-breast-w_lr_F.h5", "Breast-w"),
        Project3.load_h5_landscape("train data/05-credit-a_rf_F.h5", "Credit-a"),
        Project3.load_h5_landscape("train data/08-letter-r_knn_F.h5", "Letter-r"),
        Project3.create_triangle_landscape(16, 1.0, 4; name="Triangle", precompute=true)
    ]
end

function load_unseen_landscapes(args::Vector{String})
    fs_files_arg = parse_arg(args, "--fs-files", "")
    fs_files = filter(!isempty, split(fs_files_arg, ","))
    if isempty(fs_files)
        error("In unseen mode you must provide --fs-files=file1.h5,file2.h5")
    end

    landscapes = Project3.Landscape[]
    for fp in fs_files
        path = String(fp)
        nm = splitext(basename(path))[1]
        push!(landscapes, Project3.load_h5_landscape(path, nm))
    end

    # Step 6 hidden asymmetric triangle with n = 31.
    tri_n = parse(Int, parse_arg(args, "--tri-n", "31"))
    tri_name = parse_arg(args, "--tri-name", "Triangle-Unseen")
    
    # Use the specialized constructor to keep the n = 31 landscape memory-efficient.
    push!(landscapes, Project3.create_asymmetric_triangle_landscape(tri_n; name=tri_name))

    return landscapes
end

function run_all(
    landscapes,
    epsilon,
    n_runs,
    ga_params,
    nsga2_params,
    pso_params;
    do_landscape_visualization=true,
    do_behavior_visualization=true,
    behavior_use_landscape_map=true,
    behavior_run_selection::String="representative",
    scope_tag::String="train",
    params_source::String="cli",
    params_file::String="",
    manifest_path::String="",
    seed=nothing,
    seed_policy::String="unseeded-global-rng"
)
    println("Using epsilon=$epsilon, runs=$n_runs")

    if do_landscape_visualization
        println("Performing Landscape Analysis & Visualization...")
        for ls in landscapes
            if ls.n <= 22
                println("  Analyzing $(ls.name)...")
                optima = Project3.detect_local_optima(ls, epsilon)
                println("    Found $(length(optima)) local optima.")
                if ls.n <= 20
                    plot_hbm(ls, epsilon, optima, save_path="outputs/plots/hbm_$(ls.name).png")
                else
                    println("    Skipping HBM plot for $(ls.name) (n=$(ls.n) too large).")
                end
            else
                println("  Skipping exhaustive local-optima analysis for $(ls.name) (n=$(ls.n) too large).")
            end
        end
    end

    println("Starting Experiments...")
    algorithms = [
        (
            name = "GA",
            fn = (l; kw...) -> Project3.run_ga(l, epsilon; kw...),
            params = ga_params,
            control_policy = ga_params.adaptive ?
                "adaptive-p_mut: if fitness_std<0.001 then p_mut=min(0.5,base*5)" :
                "none"
        ),
        (
            name = "NSGA-II",
            fn = (l; kw...) -> Project3.run_nsga2(l; epsilon=epsilon, kw...),
            params = nsga2_params,
            control_policy = nsga2_params.adaptive ?
                "adaptive-(p_mut,p_cross): if MO front collapses (front_size<3 or front_diversity<0.02) then p_mut=min(0.5,base*5), p_cross=max(0.5,base*0.8)" :
                "none"
        ),
        (
            name = "PSO",
            fn = (l; kw...) -> Project3.run_pso(l, epsilon; kw...),
            params = pso_params,
            control_policy = pso_params.adaptive ?
                "adaptive-(w,c1,c2): linear schedule from start to end values across generations" :
                "none"
        )
    ]

    for ls in landscapes
        println("Landscape: $(ls.name)")
        budget = problem_budget(ls.n)
        println("  Budget rule: B(n=$(ls.n)) = $budget (pop_size=n_gens)")
        for algo in algorithms
            println("  Algorithm: $(algo.name)")
            effective_params = merge(algo.params, (pop_size=budget, n_gens=budget))
            res = Project3.run_repeated_experiments(ls, algo.fn, n_runs; epsilon=epsilon, effective_params...)
            Project3.save_results(
                res,
                ls,
                algo.name;
                scope=scope_tag,
                params_source=params_source,
                params_file=params_file,
                manifest_path=manifest_path,
                algo_params=effective_params,
                parameter_control_policy=algo.control_policy,
                seed=seed,
                seed_policy=seed_policy
            )

            if do_behavior_visualization
                run_best = [
                    maximum([
                        maximum([Project3.evaluate_fitness(ls, ind, epsilon) for ind in gen_pop])
                        for gen_pop in run_pop_history
                    ])
                    for run_pop_history in res.pop_histories
                ]
                selected_idx = if behavior_run_selection == "best"
                    argmax(run_best)
                elseif behavior_run_selection == "representative"
                    argmin(abs.(run_best .- res.mean_best))
                else
                    error("Unknown behavior_run_selection='$behavior_run_selection'. Supported: representative, best")
                end
                selected_history = res.pop_histories[selected_idx]

                Project3.plot_behavior(
                    ls,
                    epsilon,
                    selected_history;
                    all_population_histories=res.pop_histories,
                    algo_name=algo.name,
                    use_landscape_map=behavior_use_landscape_map,
                    save_path="outputs/plots/behavior_$(ls.name)_$(algo.name).png"
                )
                Project3.plot_convergence_aggregated(
                    res.histories;
                    title="$(algo.name) on $(ls.name) (aggregated across runs)",
                    save_path="outputs/plots/convergence_$(ls.name)_$(algo.name).png"
                )
                Project3.write_visit_trace(
                    "outputs/logs/visit_trace_$(ls.name)_$(algo.name).txt",
                    ls,
                    epsilon,
                    selected_history;
                    algo_name=algo.name,
                    selected_run_index=selected_idx,
                    run_selection=behavior_run_selection
                )
            end
        end
    end
end

println("Mode: $mode")
if mode == "train"
    println("Loading training landscapes...")
    landscapes = load_train_landscapes()
    if !isempty(save_params_file)
        save_params_snapshot(save_params_file, mode, epsilon, ga_params, nsga2_params, pso_params, landscapes)
    end
    manifest_path = write_run_manifest(
        mode,
        landscapes,
        epsilon,
        ga_params,
        nsga2_params,
        pso_params;
        params_source=params_source,
        params_file=params_file,
        seed=seed,
        seed_policy=seed_policy
    )
    run_all(
        landscapes,
        epsilon,
        n_runs,
        ga_params,
        nsga2_params,
        pso_params;
        do_landscape_visualization=true,
        do_behavior_visualization=true,
        behavior_use_landscape_map=true,
        behavior_run_selection="representative",
        scope_tag="train",
        params_source=params_source,
        params_file=params_file,
        manifest_path=manifest_path,
        seed=seed,
        seed_policy=seed_policy
    )
elseif mode == "unseen"
    println("Loading unseen landscapes...")
    landscapes = load_unseen_landscapes(args)
    if !isempty(save_params_file)
        save_params_snapshot(save_params_file, mode, epsilon, ga_params, nsga2_params, pso_params, landscapes)
    end
    manifest_path = write_run_manifest(
        mode,
        landscapes,
        epsilon,
        ga_params,
        nsga2_params,
        pso_params;
        params_source=params_source,
        params_file=params_file,
        seed=seed,
        seed_policy=seed_policy
    )
    run_all(
        landscapes,
        epsilon,
        n_runs,
        ga_params,
        nsga2_params,
        pso_params;
        do_landscape_visualization=false,
        do_behavior_visualization=unseen_behavior_plots,
        behavior_use_landscape_map=false,
        behavior_run_selection="best",
        scope_tag="unseen",
        params_source=params_source,
        params_file=params_file,
        manifest_path=manifest_path,
        seed=seed,
        seed_policy=seed_policy
    )
else
    error("Unknown mode '$mode'. Supported modes: train, unseen")
end

println("All tasks completed successfully!")
