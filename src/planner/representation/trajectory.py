from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import numpy as np

# We assume these are defined in the workspace as requested.
# If WorldConfig is needed strictly for type hints without circular imports:
if TYPE_CHECKING:
    from planner.environment.world import WorldConfig

from planner.geometry.bspline import BSplineCurve, create_trajectory_bspline


@dataclasses.dataclass
class TrajectoryConfig:
    """
    Configuration for the B-spline trajectory representation.
    """
    n_control_points: int
    dim: int
    start: np.ndarray
    goal: np.ndarray
    t_max: float
    degree: int = 5
    optimize_time: bool = False

    def __post_init__(self) -> None:
        """Validate the configuration parameters."""
        if self.n_control_points < self.degree + 1:
            raise ValueError(f"Need at least {self.degree + 1} control points for degree {self.degree}.")
        if self.start.shape != (self.dim,):
            raise ValueError(f"Start point shape must be ({self.dim},).")
        if self.goal.shape != (self.dim,):
            raise ValueError(f"Goal point shape must be ({self.dim},).")
        if self.t_max <= 0:
            raise ValueError("t_max must be positive.")

    @property
    def n_free_points(self) -> int:
        """Number of optimizable (interior) control points."""
        return self.n_control_points - 2

    @property
    def n_position_genes(self) -> int:
        """Number of decision variables for positions."""
        return self.n_free_points * self.dim

    @property
    def n_time_genes(self) -> int:
        """Number of decision variables for time segment durations."""
        if self.optimize_time:
            # Number of non-zero knots / time segments in the knot vector typically depends on formulation.
            # As per instructions: n_control_points - degree (number of B-spline segments)
            return self.n_control_points - self.degree
        return 0

    @property
    def n_genes(self) -> int:
        """Total number of decision variables."""
        return self.n_position_genes + self.n_time_genes


@dataclasses.dataclass
class Trajectory:
    """
    Represents a trajectory with its configuration and decision variables (genes).
    """
    config: TrajectoryConfig
    genes: np.ndarray

    def __post_init__(self) -> None:
        """Validate the genes array."""
        if self.genes.shape != (self.config.n_genes,):
            raise ValueError(f"Genes shape must be ({self.config.n_genes},), got {self.genes.shape}.")

    @property
    def free_control_points(self) -> np.ndarray:
        """
        Extract the free (interior) control points from the genes.
        Returns array of shape (n_free_points, dim).
        """
        pos_genes = self.genes[:self.config.n_position_genes]
        return pos_genes.reshape((self.config.n_free_points, self.config.dim))

    @property
    def control_points(self) -> np.ndarray:
        """
        Reconstruct the full control points array including fixed start and goal.
        Returns array of shape (n_control_points, dim).
        """
        cps = np.zeros((self.config.n_control_points, self.config.dim))
        cps[0] = self.config.start
        cps[-1] = self.config.goal
        if self.config.n_free_points > 0:
            cps[1:-1] = self.free_control_points
        return cps

    @property
    def time_variables(self) -> np.ndarray | None:
        """
        Extract the raw time variables from the genes, if optimized.
        """
        if not self.config.optimize_time:
            return None
        return self.genes[self.config.n_position_genes:]

    def normalized_time_segments(self) -> np.ndarray | None:
        """
        Get the normalized time segments (summing to 1.0) using absolute values.
        """
        t_vars = self.time_variables
        if t_vars is None:
            return None
        # Use absolute values and normalize
        abs_t = np.abs(t_vars)
        sum_t = np.sum(abs_t)
        if sum_t < 1e-9:
            # Fallback to uniform distribution if all are effectively zero
            return np.ones_like(abs_t) / len(abs_t)
        return abs_t / sum_t

    def to_bspline(self) -> BSplineCurve:
        """
        Create a BSplineCurve from the trajectory.
        """
        # The B-spline operates in parametric domain [0, 1].
        # Physical time scaling is handled at the sampling/evaluation layer.
        return create_trajectory_bspline(
            control_points=self.control_points,
            degree=self.config.degree,
        )


def create_straight_line_trajectory(config: TrajectoryConfig) -> Trajectory:
    """
    Create a trajectory initialized with control points linearly interpolated
    between start and goal.
    """
    # Create linearly interpolated interior points
    n_cp = config.n_control_points
    
    # Compute fractions for interpolation
    fractions = np.linspace(0, 1, n_cp)[1:-1]
    
    # Shape of fractions: (n_free_points, 1)
    fractions = fractions.reshape(-1, 1)
    
    # Linearly interpolate
    free_cps = config.start + fractions * (config.goal - config.start)
    
    # Flatten to position genes
    pos_genes = free_cps.flatten()
    
    if config.optimize_time:
        # Uniform time genes
        time_genes = np.ones(config.n_time_genes)
        genes = np.concatenate([pos_genes, time_genes])
    else:
        genes = pos_genes
        
    return Trajectory(config=config, genes=genes)


def create_perturbed_trajectory(
    config: TrajectoryConfig, 
    rng: np.random.Generator, 
    scale: float = 0.5
) -> Trajectory:
    """
    Create a straight-line trajectory and add Gaussian perturbation to interior control points.
    """
    # Start with a straight line
    base_traj = create_straight_line_trajectory(config)
    genes = base_traj.genes.copy()
    
    # Perturb position genes
    pos_genes = genes[:config.n_position_genes]
    pos_genes += rng.normal(loc=0.0, scale=scale, size=config.n_position_genes)
    
    if config.optimize_time:
        # Perturb time genes slightly
        time_genes = genes[config.n_position_genes:]
        # Add small perturbation, e.g. 0.1 std dev
        time_genes += rng.normal(loc=0.0, scale=0.1, size=config.n_time_genes)
        # Combine back
        genes = np.concatenate([pos_genes, time_genes])
    else:
        genes = pos_genes
        
    return Trajectory(config=config, genes=genes)


def genes_to_trajectory(genes: np.ndarray, config: TrajectoryConfig) -> Trajectory:
    """
    Create a Trajectory from a raw gene vector.
    """
    return Trajectory(config=config, genes=genes.copy())


def repair_trajectory(trajectory: Trajectory, world_config: Any) -> Trajectory:
    """
    Repair a trajectory by clamping free control points to environment bounds
    and ensuring time variables are valid.
    world_config is expected to have 'lower_bounds' and 'upper_bounds' arrays.
    """
    config = trajectory.config
    genes = trajectory.genes.copy()
    
    # 1. Clamp position genes
    if config.n_position_genes > 0:
        pos_genes = genes[:config.n_position_genes].reshape((config.n_free_points, config.dim))
        
        lower_bounds = np.asarray(world_config.lower_bounds)
        upper_bounds = np.asarray(world_config.upper_bounds)
        
        # Clip to bounds
        pos_genes = np.clip(pos_genes, lower_bounds, upper_bounds)
        
        # Flatten back
        genes[:config.n_position_genes] = pos_genes.flatten()
        
    # 2. Fix time genes if applicable
    if config.optimize_time:
        time_genes = genes[config.n_position_genes:]
        # Ensure positivity (clamp to small epsilon)
        time_genes = np.clip(time_genes, a_min=1e-6, a_max=None)
        genes[config.n_position_genes:] = time_genes
        
    return Trajectory(config=config, genes=genes)
