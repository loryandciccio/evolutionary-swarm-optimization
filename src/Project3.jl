module Project3

using HDF5, Statistics, Random, LinearAlgebra, Distributed
using Plots, CairoMakie, StatsBase, CSV, DataFrames, ProgressMeter
using Serialization

include("parser.jl")
include("fitness.jl")
include("landscape_analysis.jl")
include("visualization.jl")
include("utils.jl")

# Load algorithm implementations.
include("algorithms/genetic_algorithm.jl")
include("algorithms/nsga2.jl")
include("algorithms/pso.jl")

export load_h5_landscape, create_triangle_landscape, create_asymmetric_triangle_landscape, triangle_value, evaluate_fitness, detect_local_optima
export run_ga, run_nsga2, run_pso
export plot_hbm, plot_convergence, plot_convergence_aggregated, plot_behavior, write_visit_trace
export run_repeated_experiments, save_results, feature_subset_from_dec

end # module Project3
