# Self-contained, generated-data demonstration of the actual optimizer source.
# No external dataset, package, network request or credential is required.
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
function generated_triangle_value(k::Int, m::Float64, s::Int)
 if k == 0
  return 0.0
 end
 level = cld(k,s)
 if isodd(level)
  return m * (k % s == 0 ? s : k % s)
 end
 return m * (level*s - k)
end
n=8; m=1.0; s=2; epsilon=0.05
landscape=Landscape{Float64}(n,[generated_triangle_value(count_ones(i),m,s) for i in 0:(2^n-1)],"Synthetic-Triangle-Demo",true,m,s)
Random.seed!(20260420)
ga=run_ga(landscape,epsilon;pop_size=16,n_gens=12,adaptive=false)
nsga=run_nsga2(landscape;pop_size=16,n_gens=12,epsilon=epsilon,adaptive=false)
pso=run_pso(landscape,epsilon;pop_size=16,n_gens=12,adaptive=false)
@assert all(length(r.history)==12 for r in (ga,nsga,pso))
@assert all(isfinite,(ga.best_fitness,nsga.representative.scalar_fitness,pso.best_fitness))
println("SYNTHETIC ONLY (seed=20260420; n=8, m=1, s=2; pop=16, gens=12)")
println("GA best fitness: ",ga.best_fitness)
println("NSGA-II representative scalar fitness: ",nsga.representative.scalar_fitness)
println("PSO best fitness: ",pso.best_fitness)
