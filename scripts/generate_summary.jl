using CSV, DataFrames, Glob

const TRAIN_LANDSCAPES = Set(["Breast-w", "Credit-a", "Letter-r", "Triangle"])
const ALLOWED_SCOPES = Set(["train", "unseen"])
const ALLOWED_ALGOS = Set(["GA", "NSGA-II", "PSO"])

function parse_landscape_and_algorithm(file_path::String)
    stem = splitext(basename(file_path))[1]
    for algo in ("NSGA-II", "GA", "PSO")
        suffix = "_" * algo
        if endswith(stem, suffix)
            landscape = stem[1:(end - length(suffix))]
            return landscape, algo
        end
    end
    error("Cannot infer algorithm name from metrics filename: $file_path")
end

function parse_scope(args::Vector{String})
    for a in args
        if startswith(a, "--scope=")
            val = lowercase(String(split(a, "=", limit=2)[2]))
            if val in ("train", "unseen", "all")
                return val
            end
            error("Invalid --scope value '$val'. Allowed values: train, unseen, all")
        end
    end
    return "train"  # Default to the Step 5 training comparison scope.
end

function include_landscape_for_scope(landscape::AbstractString, scope::AbstractString)
    landscape_s = String(landscape)
    scope_s = String(scope)
    is_train = landscape_s in TRAIN_LANDSCAPES
    if scope_s == "train"
        return is_train
    elseif scope_s == "unseen"
        return !is_train
    elseif scope_s == "all"
        return true
    end
    error("Unsupported scope: $scope_s")
end

function warn_skip(file_path::String, reason::String)
    @warn "Skipping metrics file for summary generation" file=file_path reason=reason
end

function legacy_parse_metadata(file_path::String)
    try
        landscape, algo = parse_landscape_and_algorithm(file_path)
        scope = (landscape in TRAIN_LANDSCAPES) ? "train" : "unseen"
        return (landscape, algo, scope)
    catch err
        warn_skip(file_path, "missing explicit metadata and legacy filename parsing failed: $(err)")
        return nothing
    end
end

function read_metadata(df::DataFrame, file_path::String)
    has_meta = hasproperty(df, :landscape_name) && hasproperty(df, :algorithm_name) && hasproperty(df, :scope)
    if has_meta
        landscape = string(df.landscape_name[1])
        algo = string(df.algorithm_name[1])
        scope = lowercase(strip(string(df.scope[1])))

        if isempty(strip(landscape))
            warn_skip(file_path, "empty landscape_name metadata")
            return nothing
        end
        if !(algo in ALLOWED_ALGOS)
            warn_skip(file_path, "invalid algorithm_name metadata '$algo'")
            return nothing
        end
        if !(scope in ALLOWED_SCOPES)
            warn_skip(file_path, "invalid scope metadata '$scope'")
            return nothing
        end
        return (landscape, algo, scope)
    end
    return legacy_parse_metadata(file_path)
end

function generate_summary_table(scope::String="train")
    files = glob("outputs/metrics/*.csv")
    results = []
    
    for f in files
        df = CSV.read(
            f,
            DataFrame;
            types=Dict(
                :best_bitstring => String
            )
        )
        metadata = read_metadata(df, f)
        if metadata === nothing
            continue
        end
        landscape, algo, file_scope = metadata
        if scope != "all" && file_scope != scope
            continue
        end
        # Keep a compatibility guard for older CSV files that may lack explicit scope metadata.
        if scope == "train" && !include_landscape_for_scope(landscape, "train") && !hasproperty(df, :scope)
            warn_skip(f, "legacy file inferred as non-train while train scope requested")
            continue
        end
        
        push!(results, (
            Landscape = landscape,
            Algorithm = algo,
            MeanBest = df.mean_best[1],
            StdBest = df.std_best[1],
            MaxBest = df.max_best[1],
            BestInd = hasproperty(df, :best_ind) ? string(df.best_ind[1]) : "",
            BestBitstring = hasproperty(df, :best_bitstring) ? string(df.best_bitstring[1]) : ""
        ))
    end
    
    summary_df = isempty(results) ? DataFrame(
        Landscape = String[],
        Algorithm = String[],
        MeanBest = Float64[],
        StdBest = Float64[],
        MaxBest = Float64[],
        BestInd = String[],
        BestBitstring = String[]
    ) : DataFrame(results)
    # Sort the final table for readability.
    sort!(summary_df, [:Landscape, :Algorithm])
    
    mkpath("outputs/tables")
    csv_path = "outputs/tables/summary_results_$(scope).csv"
    CSV.write(csv_path, summary_df)
    println("Summary scope: $scope")
    println("CSV saved to: $csv_path")
    println(summary_df)
    
    return summary_df
end

function write_final_summary(summary_df::DataFrame, scope::String)
    mkpath("outputs")
    md_path = "outputs/summary_$(scope).md"
    
    lines = String[]
    push!(lines, "# Project 3 Final Summary ($(uppercase(scope))) - Bio-inspired Algorithms (IT3708)")
    push!(lines, "")
    push!(lines, "This summary is generated directly from `outputs/tables/summary_results_$(scope).csv`.")
    push!(lines, "")
    push!(lines, "## Results")
    push!(lines, "")
    push!(lines, "| Landscape | Algorithm | MeanBest | StdBest | MaxBest | BestInd | BestBitstring |")
    push!(lines, "|---|---|---:|---:|---:|---:|---|")
    
    for row in eachrow(summary_df)
        push!(lines, "| $(row.Landscape) | $(row.Algorithm) | $(row.MeanBest) | $(row.StdBest) | $(row.MaxBest) | $(row.BestInd) | $(row.BestBitstring) |")
    end
    
    content = join(lines, "\n") * "\n"
    write(md_path, content)
    println("Markdown summary saved to: $md_path")
end

scope = parse_scope(ARGS)
summary_df = generate_summary_table(scope)
write_final_summary(summary_df, scope)
