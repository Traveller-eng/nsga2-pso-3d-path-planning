from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from planner.environment.world import WorldConfig
from planner.representation.trajectory import Trajectory, TrajectoryConfig, genes_to_trajectory
from planner.geometry.sampling import compute_sample_count, sample_trajectory, parametric_to_physical
from planner.evaluation.objectives import compute_objectives
from planner.evaluation.constraints import compute_total_violation, is_feasible
from planner.environment.obstacles import compute_min_distance


@dataclass
class EvaluationResult:
    """
    Result of evaluating a trajectory.
    """
    objectives: np.ndarray
    total_violation: float
    violation_details: dict[str, float]
    is_feasible: bool
    n_samples: int
    min_clearance: float
    max_speed: float
    max_accel: float
    path_length: float
    raw_snap: float


class TrajectoryEvaluator:
    """
    Unified evaluator for trajectories calculating objectives and constraints.
    """
    
    def __init__(self, world: WorldConfig):
        """
        Initialize the evaluator with a world configuration.
        """
        self.world = world
        self.n_samples = compute_sample_count(self.world.v_max, self.world.d_safe, self.world.t_max)
        
    def evaluate(self, trajectory: Trajectory) -> EvaluationResult:
        """
        Evaluate a given trajectory for objectives and constraints.
        """
        curve = trajectory.to_bspline()
        param_samples = sample_trajectory(curve, self.n_samples)
        physical_samples = parametric_to_physical(param_samples, self.world.t_max)
        
        positions = physical_samples['positions']
        velocity = physical_samples['velocity']
        acceleration = physical_samples['acceleration']
        snap = physical_samples['snap']
        times = physical_samples.get('t')
        
        objectives, raw_snap = compute_objectives(
            positions=positions,
            snap_samples=snap,
            obstacles=self.world.obstacles,
            d_safe=self.world.d_safe,
            t_max=self.world.t_max,
            d_comfort=self.world.d_comfort,
            times=times
        )
        
        total_violation, violation_details = compute_total_violation(
            positions=positions,
            velocity=velocity,
            acceleration=acceleration,
            obstacles=self.world.obstacles,
            d_safe=self.world.d_safe,
            v_max=self.world.v_max,
            a_max=self.world.a_max,
            lower_bounds=self.world.lower_bounds,
            upper_bounds=self.world.upper_bounds,
            times=times
        )
        
        feasible = is_feasible(total_violation)
        
        # Calculate diagnostics
        if len(self.world.obstacles) > 0:
            min_clearance = min([compute_min_distance(positions[i], self.world.obstacles, 0.0 if times is None else times[i]) for i in range(self.n_samples)])
        else:
            min_clearance = float('inf')
            
        max_speed = float(np.max(np.linalg.norm(velocity, axis=1)))
        max_accel = float(np.max(np.linalg.norm(acceleration, axis=1)))
        path_length = float(objectives[0])
        
        return EvaluationResult(
            objectives=objectives,
            total_violation=total_violation,
            violation_details=violation_details,
            is_feasible=feasible,
            n_samples=self.n_samples,
            min_clearance=min_clearance,
            max_speed=max_speed,
            max_accel=max_accel,
            path_length=path_length,
            raw_snap=raw_snap
        )

    def evaluate_from_genes(self, genes: np.ndarray, config: TrajectoryConfig) -> EvaluationResult:
        """
        Evaluate a trajectory represented by a genes array.
        """
        trajectory = genes_to_trajectory(genes, config)
        return self.evaluate(trajectory)
