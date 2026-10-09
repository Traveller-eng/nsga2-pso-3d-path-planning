from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Union


@dataclass
class SphereObstacle:
    """Spherical obstacle representation."""
    center: np.ndarray
    radius: float
    velocity: np.ndarray | None = None
    path: list[np.ndarray] | None = None
    path_times: list[float] | None = None

    def position_at(self, t: float) -> np.ndarray:
        """Calculate the center position of the obstacle at time t."""
        if self.path is not None:
            if self.path_times is None:
                raise ValueError("SphereObstacle.path requires path_times to define the interpolation schedule.")
            if len(self.path) != len(self.path_times):
                raise ValueError("SphereObstacle.path and path_times must have the same length.")
            if t <= self.path_times[0]:
                return np.asarray(self.path[0], dtype=float)
            if t >= self.path_times[-1]:
                return np.asarray(self.path[-1], dtype=float)
            for idx in range(len(self.path_times) - 1):
                t0 = self.path_times[idx]
                t1 = self.path_times[idx + 1]
                if t0 <= t <= t1:
                    p0 = np.asarray(self.path[idx], dtype=float)
                    p1 = np.asarray(self.path[idx + 1], dtype=float)
                    alpha = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                    return p0 + alpha * (p1 - p0)
            return np.asarray(self.path[-1], dtype=float)
        if self.velocity is not None:
            return self.center + self.velocity * t
        return self.center

    def distance_to_point(self, point: np.ndarray, t: float = 0.0) -> float:
        """
        Calculate signed Euclidean distance from a point to the obstacle surface.
        Positive value means outside, negative means inside.
        """
        pos = self.position_at(t)
        dist = np.linalg.norm(point - pos)
        return float(dist - self.radius)

    def is_collision(self, point: np.ndarray, d_safe: float, t: float = 0.0) -> bool:
        """Check if the distance to the point is less than the safe distance."""
        return self.distance_to_point(point, t) < d_safe


@dataclass
class AABBObstacle:
    """Axis-Aligned Bounding Box (AABB) obstacle representation."""
    lower: np.ndarray
    upper: np.ndarray
    velocity: np.ndarray | None = None

    def position_at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the lower and upper bounds of the AABB at time t."""
        if self.velocity is not None:
            offset = self.velocity * t
            return self.lower + offset, self.upper + offset
        return self.lower, self.upper

    def distance_to_point(self, point: np.ndarray, t: float = 0.0) -> float:
        """
        Calculate signed distance from a point to the AABB surface.
        Positive value means outside, negative means inside.
        """
        current_lower, current_upper = self.position_at(t)
        
        # Clamp point to the box to find the closest point on/in the box
        clamped_point = np.clip(point, current_lower, current_upper)
        
        # If point is outside the box, the distance is the distance to the clamped point
        if not np.array_equal(point, clamped_point):
            return float(np.linalg.norm(point - clamped_point))
        
        # If point is inside, distance to surface is the minimum distance to any face
        dist_to_lower = point - current_lower
        dist_to_upper = current_upper - point
        min_dist = np.min(np.minimum(dist_to_lower, dist_to_upper))
        return float(-min_dist)

    def is_collision(self, point: np.ndarray, d_safe: float, t: float = 0.0) -> bool:
        """Check if the distance to the point is less than the safe distance."""
        return self.distance_to_point(point, t) < d_safe


# Type alias for any obstacle
Obstacle = Union[SphereObstacle, AABBObstacle]


def compute_min_distance(point: np.ndarray, obstacles: list[Obstacle], t: float = 0.0) -> float:
    """
    Compute the minimum distance from a point to the surface of any obstacle in the list.
    Returns float('inf') if there are no obstacles.
    """
    if not obstacles:
        return float('inf')
    
    return min((obs.distance_to_point(point, t) for obs in obstacles))


def check_collision(point: np.ndarray, obstacles: list[Obstacle], d_safe: float, t: float = 0.0) -> bool:
    """
    Check if the point is in collision (or within safe distance) of any obstacle.
    """
    if not obstacles:
        return False
    
    return any(obs.is_collision(point, d_safe, t) for obs in obstacles)
