# Hybrid NSGA-II + PSO for Constrained Dynamic 3D Trajectory Optimization

Research-grade implementation of a hybrid multi-objective optimization approach
for 3D UAV trajectory planning with obstacle avoidance.

## Overview

- **Trajectory**: Degree-5 clamped B-spline in 3D
- **Objectives**: Path length, clearance risk, integrated squared snap — all minimized
- **Constraints**: Collision, velocity, acceleration, environment bounds
- **Optimization**: NSGA-II + Tchebycheff-decomposed PSO with Deb-style constrained domination

## Installation

```bash
pip install -e ".[dev]"
```

## Testing

```bash
pytest
```

## Dynamic Warm-Start Experiment

Run the paired cold-start versus warm-start NSGA-II experiment with the frozen
scenario, 10 paired seeds, four event times, and 100 objective evaluations per
event:

```bash
python scripts/run_dynamic_experiment.py --config experiments/configs/dynamic_warm_start.json
```

The run writes `dynamic_warm_start_runs.json`, `dynamic_warm_start_summary.json`,
`dynamic_warm_start_analysis.json`, and `dynamic_warm_start_summary.csv` under
`experiments/results/`. Paired seeds are `11, 22, 33, 44, 55, 66, 77, 88, 99,
110`. HV and IGD use fixed objective scales and the same
event-specific pooled feasible reference front across both conditions and all
seeds. The scenario is a single bounded clearance/violation stress case; its
fixed witness is infeasible at every event, so results do not establish general
algorithm superiority or test a feasibility transition.

## Project Structure

```
src/planner/
├── geometry/       # B-spline math, trajectory sampling
├── environment/    # Obstacles and world configuration
├── evaluation/     # Objectives, constraints, unified evaluator
├── optimization/   # NSGA-II, PSO, MOPSO, hybrid (future)
├── representation/ # Trajectory representation
└── metrics/        # Hypervolume, IGD (future)
```
