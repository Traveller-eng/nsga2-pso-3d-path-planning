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
