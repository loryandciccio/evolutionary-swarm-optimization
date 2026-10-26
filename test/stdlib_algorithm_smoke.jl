# Dependency-free unit smoke test of actual optimizer and fitness sources.
# Does not test HDF5 parser, plotting or experiment driver.
using Random, Statistics
struct Landscape{T<:Number}
 n::Int
 lookup_table::Vector{T}
 name::String
 is_synthetic::Bool
 triangle_m::Float64
 triangle_s::Int
end
include("../src/fitness.jl")
include("../src/algorithms/genetic_algorithm.jl")
include("../src/algorithms/nsga2.jl")
include("../src/algorithms/pso.jl")
Random.seed!(20260420)
n=8
L=Landscape{Float64}(n,[Float64(count_ones(i))/n for i in 0:(2^n-1)],"Synthetic-Bitcount",true,0.0,1)
ga=run_ga(L,0.05;pop_size=12,n_gens=8,adaptive=false)
nsga=run_nsga2(L;pop_size=12,n_gens=8,epsilon=0.05,adaptive=false)
pso=run_pso(L,0.05;pop_size=12,n_gens=8,adaptive=false)
@assert all(length(r.history)==8 for r in (ga,nsga,pso))
@assert all(isfinite,(ga.best_fitness,nsga.representative.scalar_fitness,pso.best_fitness))
@assert all(0<=v<=2^n-1 for v in (ga.best_ind,nsga.representative.ind,pso.best_ind))
println("PASS: synthetic-only GA / NSGA-II / PSO smoke test")
