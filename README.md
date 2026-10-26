# Evolutionary and Swarm Optimization in Julia

A portfolio edition of an individual academic project completed in 2026 for **IT3708 — Bio-Inspired Artificial Intelligence**, Norwegian University of Science and Technology (NTNU).

**Developed by Lorenzo Cozzolino.**

## Overview

This project explores combinatorial optimization over binary feature-selection landscapes and synthetic triangle landscapes. A solution is a bit string encoding active features. Three evolutionary/swarm optimizers are compared with common parameter-budget policies and repeated stochastic runs.

The public edition preserves the optimizer source and a small generated-data demonstration. Course-provided HDF5 lookup tables, hidden-test specifications, classroom documents, serializations and other material of uncertain redistribution status are **not distributed**. Thus the original feature-selection experiments cannot be reproduced from this repository alone.

## Algorithms implemented

| Method | Optimization | State and search operators |
| --- | --- | --- |
| Genetic Algorithm (GA) | Single objective: maximize accuracy minus feature penalty | Integer-encoded bit strings, tournament selection, one-point crossover, bit-flip mutation, elitism |
| NSGA-II | Multiobjective: minimize prediction error and feature penalty | Integer-encoded bit strings, nondominated sorting, crowding distances, elitist parent-plus-offspring survival |
| Binary Particle Swarm Optimization (PSO) | Single objective: maximize accuracy minus feature penalty | Binary particle positions, real-valued velocities, sigmoid updates, personal/global best |

Parameter-control policies in the source include increasing mutation on GA diversity collapse, adapting mutation/crossover on NSGA-II Pareto-front geometry, and scheduling PSO inertia and acceleration parameters. They are implementation choices, not claims to have invented these algorithms.

The original scalar fitness is

`f(x) = accuracy(x) - epsilon * popcount(x) / n`.

For the documented feature-selection training experiments, `epsilon = 0.05`. NSGA-II minimizes the two objectives `(1 - accuracy(x), epsilon * popcount(x) / n)` and chooses a representative Pareto-front solution using the scalar fitness. When `epsilon = 0`, the second objective becomes constant, so the search does not represent a nontrivial accuracy/sparsity Pareto tradeoff.

The original budget policy sets both population size and number of generations to `B(n) = min(n^2, 400)`. This controls nominal settings, **not** actual runtime or precisely equal numbers of fitness evaluations.

## Repository contents

- `src/algorithms/`: GA, NSGA-II, and binary PSO implementations.
- `src/parser.jl`, `src/fitness.jl`: landscape representations and objective evaluation.
- `src/landscape_analysis.jl`, `src/visualization.jl`: Hamming-1 local-optimum analysis and diagnostic visualizations.
- `src/utils.jl`: repeated experiments, aggregation, serialization and result reporting.
- `scripts/demo.jl`: seeded synthetic triangle demonstration with reduced computational budget.
- `scripts/main.jl`: historical experiment driver. Training requires omitted course-format HDF5 data; hidden-test mode is unavailable in this edition.
- `scripts/generate_summary.jl`: original metrics summarization script; requires appropriate `outputs/metrics/` files.
- `test/stdlib_algorithm_smoke.jl`: standalone Julia unit smoke test using actual optimizer/fitness implementation files and generated data.
- `results/historical_training_summary.csv`: archived aggregates, not newly validated results.
- `figures/`: selected plots from the *synthetic training Triangle* experiment.

## Installation

The standalone synthetic demo and algorithm smoke test below require **Julia only**, with no package installation. For the original experiment and visualization modules, install Julia 1.x and resolve packages from `Project.toml`:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

First-time package installation may download dependencies and compiled artifacts. This release does not ship a `Manifest.toml`, so transitive package versions are not pinned. The visualization stack includes Plots and CairoMakie and can be resource-intensive.

### Synthetic demo

```bash
julia --project=. scripts/demo.jl
```

This uses a locally generated triangular landscape with `n=8, m=1, s=2`, a fixed seed (`20260420`), population size 16 and 12 generations per optimizer. The verified Julia 1.11.7 run of this reduced demo printed fitness **1.9875** for GA, the NSGA-II scalar representative and PSO; these are **synthetic demonstration fitness** values, not real-world benchmark performance. No credential, external endpoint or non-synthetic dataset is accessed by the demonstration logic.

### Lightweight optimizer test

```bash
julia test/stdlib_algorithm_smoke.jl
```

This dependency-free unit smoke test loads the actual GA/NSGA-II/PSO source files and fitness function, using an in-memory generated bit-count landscape. It does not test the HDF5 parser, plot generation or full integration. The test can be run with Julia even before `Pkg.instantiate()`.

## Input format and provenance

The original course-provided training landscapes represented mean classifier accuracy for each nonempty feature subset. Julia's HDF5 interface reads a two-dimensional `accuracies` dataset and averages over repeated evaluations. Bit strings are reported with the most significant bit on the left, while internal operators use bit indices for search.

Original upstream dataset families:

- [UCI Breast Cancer Wisconsin (Original)](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original), 9 features.
- [UCI Credit Approval](https://archive.ics.uci.edu/dataset/27/credit+approval), 15 features.
- [UCI Letter Recognition](https://archive.ics.uci.edu/dataset/59/letter+recognition), 16 features.

UCI lists the original datasets under CC BY 4.0 subject to attribution, but that does not establish redistribution rights for the **course-generated HDF5 accuracy tables**. The files are excluded. Downloading UCI raw datasets does **not** recreate these lookup tables, as the original feature-selection preprocessing, models and repeated accuracy runs would also be required.

The historic full experiment additionally included a generated training Triangle (`n=16, m=1, s=4`) and held-back test instances. The test inputs, associated results and the supplied hidden synthetic landscape definition are excluded. This edition deliberately disables the course-specific hidden-landscape constructor; running the `--mode=unseen` path of `scripts/main.jl` will fail. The training workflow in `scripts/main.jl` similarly requires absent HDF5 files in the original expected directory layout.

## Historical evaluation and results

The original documented evaluation used 50 repeated runs per algorithm and training landscape, with a top-level RNG seed `20260420`. The archived summary reports mean best fitness, standard deviation, maximum, and the best solution state for Breast-w, Credit-a, Letter-r and Triangle.

For the synthetic training Triangle only, the archived maximum scalar fitness was **3.9875** for GA, NSGA-II and PSO; each archived standard deviation was **0.0**. These are historical observations, not newly reproduced claims or proof of equivalent search behavior. Selected triangle plots are in `figures/`.

The feature-selection results in `results/historical_training_summary.csv` were produced using course-provided lookup tables that are not included here. They therefore are **not independently reproducible** from this public edition and were not reevaluated on an independent held-out raw-data split for this release. Optimizing over precomputed accuracy tables is not itself evidence of generalization on new patients, credit applicants or character images.

## Limitations

- Full benchmark integration, plotting and HDF5 parsing require the omitted original inputs and installed Julia packages.
- Source and demonstration are educational; no production-ready claims, new benchmarks or guaranteed reproducibility are made.
- Runtime and algorithm comparison should account for fitness-evaluation budgets, seeds, variance and package-version differences.
- The small synthetic demo intentionally exercises a different, far cheaper setting than the historical runs.
- The historical hidden-test answer files and full serialized results are absent by design.

## Research attribution and licensing

This is an implementation and experimental comparison of existing approaches. Foundational references include David E. Goldberg, *Genetic Algorithms in Search, Optimization and Machine Learning* (1989); Kalyanmoy Deb et al., *A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II* (IEEE TEC, 2002); and James Kennedy and Russell Eberhart, *Particle Swarm Optimization* (ICNN, 1995), with later binary PSO adaptations.

**License status:** The portfolio edition does not grant a new open-source license. No MIT/Apache license has been assumed, and excluded third-party academic materials are not relicensed. Publication of the source and redistribution permissions for the excluded materials are distinct decisions.
