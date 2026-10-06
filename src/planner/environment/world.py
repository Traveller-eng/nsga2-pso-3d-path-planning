from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .obstacles import Obstacle, SphereObstacle, AABBObstacle


@dataclass
class WorldConfig:
    """Configuration for the trajectory optimization world environment."""
    dim: int
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    obstacles: list[Obstacle]
    d_safe: float
    d_comfort: float
    v_max: float
    a_max: float
    t_max: float
    start: np.ndarray
    goal: np.ndarray

    def is_in_bounds(self, point: np.ndarray) -> bool:
        """Check if a point is within the environment bounds."""
        return bool(np.all(point >= self.lower_bounds) and np.all(point <= self.upper_bounds))

    def straight_line_distance(self) -> float:
        """Calculate the Euclidean distance from the start position to the goal position."""
        return float(np.linalg.norm(self.goal - self.start))

    def estimate_t_max(self, factor: float = 1.75) -> float:
        """
        Estimate a reasonable total trajectory time based on the straight-line distance,
        maximum velocity, and a heuristic scaling factor.
        """
        return float(factor * self.straight_line_distance() / self.v_max)


def make_2d_static_world(
    bounds_max: tuple[float, float] = (10.0, 10.0),
    start: tuple[float, float] = (0.0, 0.0),
    goal: tuple[float, float] = (9.0, 9.0),
    obstacles: Optional[list[Obstacle]] = None,
    v_max: float = 3.0,
    a_max: float = 5.0,
    d_safe: float = 0.5,
    d_comfort: float = 1.5,
    t_max_factor: float = 1.75
) -> WorldConfig:
    """Factory to create a 2D static test environment."""
    if obstacles is None:
        obstacles = [
            SphereObstacle(center=np.array([5.0, 5.0]), radius=1.5),
            AABBObstacle(lower=np.array([2.0, 7.0]), upper=np.array([4.0, 8.0]))
        ]
        
    start_arr = np.array(start, dtype=float)
    goal_arr = np.array(goal, dtype=float)
    
    world = WorldConfig(
        dim=2,
        lower_bounds=np.zeros(2, dtype=float),
        upper_bounds=np.array(bounds_max, dtype=float),
        obstacles=obstacles,
        d_safe=d_safe,
        d_comfort=d_comfort,
        v_max=v_max,
        a_max=a_max,
        t_max=0.0,  # placeholder
        start=start_arr,
        goal=goal_arr
    )
    
    # Estimate reasonable t_max based on distance and velocity
    world.t_max = world.estimate_t_max(factor=t_max_factor)
    
    return world


def make_3d_static_world(
    bounds_max: tuple[float, float, float] = (10.0, 10.0, 10.0),
    start: tuple[float, float, float] = (0.0, 0.0, 0.0),
    goal: tuple[float, float, float] = (9.0, 9.0, 9.0),
    obstacles: Optional[list[Obstacle]] = None,
    v_max: float = 3.0,
    a_max: float = 5.0,
    d_safe: float = 0.5,
    d_comfort: float = 1.5,
    t_max_factor: float = 1.75
) -> WorldConfig:
    """Factory to create a 3D static test environment."""
    if obstacles is None:
        obstacles = [
            SphereObstacle(center=np.array([5.0, 5.0, 5.0]), radius=2.0),
            AABBObstacle(lower=np.array([2.0, 7.0, 2.0]), upper=np.array([4.0, 8.0, 6.0]))
        ]
        
    start_arr = np.array(start, dtype=float)
    goal_arr = np.array(goal, dtype=float)
    
    world = WorldConfig(
        dim=3,
        lower_bounds=np.zeros(3, dtype=float),
        upper_bounds=np.array(bounds_max, dtype=float),
        obstacles=obstacles,
        d_safe=d_safe,
        d_comfort=d_comfort,
        v_max=v_max,
        a_max=a_max,
        t_max=0.0,  # placeholder
        start=start_arr,
        goal=goal_arr
    )
    
    world.t_max = world.estimate_t_max(factor=t_max_factor)
    
    return world


def make_3d_dynamic_world(
    bounds_max: tuple[float, float, float] = (10.0, 10.0, 10.0),
    start: tuple[float, float, float] = (0.0, 0.0, 0.0),
    goal: tuple[float, float, float] = (9.0, 9.0, 9.0),
    obstacles: Optional[list[Obstacle]] = None,
    v_max: float = 3.0,
    a_max: float = 5.0,
    d_safe: float = 0.5,
    d_comfort: float = 1.5,
    t_max_factor: float = 1.75
) -> WorldConfig:
    """Factory to create a 3D dynamic test environment with moving obstacles."""
    if obstacles is None:
        obstacles = [
            SphereObstacle(
                center=np.array([5.0, 5.0, 5.0]), 
                radius=1.5, 
                velocity=np.array([0.5, -0.2, 0.1])
            ),
            AABBObstacle(
                lower=np.array([2.0, 7.0, 2.0]), 
                upper=np.array([4.0, 8.0, 6.0]),
                velocity=np.array([-0.3, 0.0, 0.2])
            ),
            SphereObstacle(center=np.array([7.0, 3.0, 8.0]), radius=1.0) # static obstacle
        ]
        
    start_arr = np.array(start, dtype=float)
    goal_arr = np.array(goal, dtype=float)
    
    world = WorldConfig(
        dim=3,
        lower_bounds=np.zeros(3, dtype=float),
        upper_bounds=np.array(bounds_max, dtype=float),
        obstacles=obstacles,
        d_safe=d_safe,
        d_comfort=d_comfort,
        v_max=v_max,
        a_max=a_max,
        t_max=0.0,  # placeholder
        start=start_arr,
        goal=goal_arr
    )
    
    world.t_max = world.estimate_t_max(factor=t_max_factor)
    
    return world
