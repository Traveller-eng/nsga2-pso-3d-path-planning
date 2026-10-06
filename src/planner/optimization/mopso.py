from __future__ import annotations

import dataclasses
import math
import numpy as np

from planner.environment.world import WorldConfig
from planner.evaluation.evaluator import EvaluationResult, TrajectoryEvaluator
from planner.representation.trajectory import (
    Trajectory,
    TrajectoryConfig,
    create_perturbed_trajectory,
    create_straight_line_trajectory,
    genes_to_trajectory,
    repair_trajectory,
)


@dataclasses.dataclass
class ObjectiveNormalization:
    """Frozen objective normalization state for a PSO phase."""

    ideal: np.ndarray
    nadir: np.ndarray
    scale: np.ndarray

    @classmethod
    def from_objectives(cls, objectives: np.ndarray, eps: float = 1e-12) -> "ObjectiveNormalization":
        if objectives.ndim != 2:
            raise ValueError("objectives must be a 2D array.")
        ideal = np.min(objectives, axis=0)
        nadir = np.max(objectives, axis=0)
        scale = np.maximum(nadir - ideal, eps)
        return cls(ideal=ideal, nadir=nadir, scale=scale)

    def normalize(self, objectives: np.ndarray) -> np.ndarray:
        return (objectives - self.ideal) / self.scale


@dataclasses.dataclass
class Particle:
    """Particle state for decomposition-based MOPSO."""

    position: np.ndarray
    velocity: np.ndarray
    trajectory: Trajectory
    evaluation: EvaluationResult
    pbest_position: np.ndarray
    pbest_evaluation: EvaluationResult
    weight: np.ndarray
    neighborhood: list[int]


@dataclasses.dataclass
class ArchiveEntry:
    trajectory: Trajectory
    evaluation: EvaluationResult
    position: np.ndarray
    crowding_distance: float = 0.0


def generate_weight_vectors(n_weights: int, n_objectives: int = 3) -> np.ndarray:
    """Generate deterministic simplex-lattice weight vectors."""
    if n_weights <= 0:
        return np.empty((0, n_objectives), dtype=float)
    if n_objectives != 3:
        raise ValueError("This MOPSO implementation currently expects three objectives.")

    divisions = 1
    while math.comb(divisions + n_objectives - 1, n_objectives - 1) < n_weights:
        divisions += 1

    weights = []
    for i in range(divisions + 1):
        for j in range(divisions - i + 1):
            k = divisions - i - j
            weights.append([i / divisions, j / divisions, k / divisions])

    weights = np.array(weights, dtype=float)
    order = np.lexsort((weights[:, 2], weights[:, 1], weights[:, 0]))
    return weights[order][:n_weights]


def tchebycheff_scalarization(
    objectives: np.ndarray,
    weight: np.ndarray,
    normalization: ObjectiveNormalization,
) -> float:
    """Weighted Tchebycheff scalarization on frozen normalized objectives."""
    normalized = normalization.normalize(objectives)
    normalized_ideal = normalization.normalize(normalization.ideal)
    return float(np.max(weight * np.abs(normalized - normalized_ideal)))


def evaluation_better_for_weight(
    candidate: EvaluationResult,
    incumbent: EvaluationResult,
    weight: np.ndarray,
    normalization: ObjectiveNormalization,
) -> bool:
    """Deb-style constrained comparison, scalarized for feasible ties."""
    if candidate.is_feasible and not incumbent.is_feasible:
        return True
    if not candidate.is_feasible and incumbent.is_feasible:
        return False
    if not candidate.is_feasible and not incumbent.is_feasible:
        return candidate.total_violation < incumbent.total_violation

    cand_value = tchebycheff_scalarization(candidate.objectives, weight, normalization)
    inc_value = tchebycheff_scalarization(incumbent.objectives, weight, normalization)
    return cand_value < inc_value


def pareto_dominates(a: EvaluationResult, b: EvaluationResult) -> bool:
    """Pareto dominance on feasible objective vectors, all minimized."""
    return bool(np.all(a.objectives <= b.objectives) and np.any(a.objectives < b.objectives))


def build_neighborhoods(weights: np.ndarray, neighborhood_size: int) -> list[list[int]]:
    """Return nearest weight-vector neighborhoods, including each particle itself."""
    if len(weights) == 0:
        return []
    size = max(1, min(neighborhood_size, len(weights)))
    neighborhoods = []
    for weight in weights:
        distances = np.linalg.norm(weights - weight, axis=1)
        neighborhoods.append(list(np.argsort(distances)[:size]))
    return neighborhoods


def calculate_archive_crowding(entries: list[ArchiveEntry]) -> None:
    """Assign NSGA-II-style crowding distance for archive pruning."""
    if not entries:
        return
    if len(entries) <= 2:
        for entry in entries:
            entry.crowding_distance = float("inf")
        return

    for entry in entries:
        entry.crowding_distance = 0.0

    n_obj = len(entries[0].evaluation.objectives)
    for m in range(n_obj):
        entries.sort(key=lambda entry: entry.evaluation.objectives[m])
        entries[0].crowding_distance = float("inf")
        entries[-1].crowding_distance = float("inf")
        f_min = entries[0].evaluation.objectives[m]
        f_max = entries[-1].evaluation.objectives[m]
        if f_max - f_min <= 1e-12:
            continue
        for i in range(1, len(entries) - 1):
            if entries[i].crowding_distance != float("inf"):
                gap = entries[i + 1].evaluation.objectives[m] - entries[i - 1].evaluation.objectives[m]
                entries[i].crowding_distance += gap / (f_max - f_min)


class ExternalArchive:
    """Feasible non-dominated archive with crowding-distance cap."""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.entries: list[ArchiveEntry] = []

    def update(self, trajectory: Trajectory, evaluation: EvaluationResult, position: np.ndarray) -> None:
        if not evaluation.is_feasible:
            return

        candidate = ArchiveEntry(trajectory=trajectory, evaluation=evaluation, position=position.copy())
        if any(pareto_dominates(entry.evaluation, candidate.evaluation) for entry in self.entries):
            return

        self.entries = [
            entry for entry in self.entries
            if not pareto_dominates(candidate.evaluation, entry.evaluation)
        ]
        self.entries.append(candidate)
        self._prune()

    def _prune(self) -> None:
        while len(self.entries) > self.max_size:
            calculate_archive_crowding(self.entries)
            finite = [entry for entry in self.entries if entry.crowding_distance != float("inf")]
            if finite:
                remove = min(finite, key=lambda entry: entry.crowding_distance)
            else:
                remove = self.entries[-1]
            self.entries = [entry for entry in self.entries if entry is not remove]


class MOPSOOptimizer:
    """Standalone decomposition-based multiobjective PSO optimizer."""

    def __init__(
        self,
        population_size: int = 100,
        max_evaluations: int = 10000,
        inertia: float = 0.5,
        cognitive: float = 1.5,
        social: float = 1.5,
        velocity_clamp: float = 0.2,
        neighborhood_size: int = 10,
        archive_size: int = 100,
        seed: int = 42,
    ):
        self.population_size = population_size
        self.max_evaluations = max_evaluations
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.velocity_clamp = velocity_clamp
        self.neighborhood_size = neighborhood_size
        self.archive = ExternalArchive(max_size=archive_size)
        self.rng = np.random.default_rng(seed)

        self.evals = 0
        self.iteration = 0
        self.history: list[dict] = []
        self.normalization: ObjectiveNormalization | None = None

    def _get_bounds(self, config: TrajectoryConfig, world: WorldConfig) -> tuple[np.ndarray, np.ndarray]:
        lower = np.tile(world.lower_bounds, config.n_free_points)
        upper = np.tile(world.upper_bounds, config.n_free_points)
        return lower, upper

    def _velocity_bounds(self, lower_bounds: np.ndarray, upper_bounds: np.ndarray) -> np.ndarray:
        return self.velocity_clamp * (upper_bounds - lower_bounds)

    def _evaluate(self, trajectory: Trajectory, evaluator: TrajectoryEvaluator) -> EvaluationResult:
        self.evals += 1
        return evaluator.evaluate(trajectory)

    def _initialize_particles(
        self,
        world: WorldConfig,
        config: TrajectoryConfig,
        evaluator: TrajectoryEvaluator,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ) -> list[Particle]:
        if self.max_evaluations <= 0 or self.population_size <= 0:
            return []

        weights = generate_weight_vectors(self.population_size, 3)
        particles: list[Particle] = []
        scale = np.linalg.norm(world.upper_bounds - world.lower_bounds) * 0.05
        velocity_limit = self._velocity_bounds(lower_bounds, upper_bounds)

        while len(particles) < self.population_size and self.evals < self.max_evaluations:
            if len(particles) == 0:
                trajectory = create_straight_line_trajectory(config)
            else:
                trajectory = create_perturbed_trajectory(config, self.rng, scale=scale)
                trajectory = repair_trajectory(trajectory, world)

            evaluation = self._evaluate(trajectory, evaluator)
            position = trajectory.genes.copy()
            velocity = self.rng.uniform(-velocity_limit, velocity_limit)
            particles.append(
                Particle(
                    position=position,
                    velocity=velocity,
                    trajectory=trajectory,
                    evaluation=evaluation,
                    pbest_position=position.copy(),
                    pbest_evaluation=evaluation,
                    weight=weights[len(particles)].copy(),
                    neighborhood=[],
                )
            )
            self.archive.update(trajectory, evaluation, position)

        active_weights = np.array([particle.weight for particle in particles])
        neighborhoods = build_neighborhoods(active_weights, self.neighborhood_size)
        for particle, neighborhood in zip(particles, neighborhoods):
            particle.neighborhood = neighborhood

        if particles:
            objectives = np.array([particle.evaluation.objectives for particle in particles])
            self.normalization = ObjectiveNormalization.from_objectives(objectives)

        return particles

    def _select_lbest(self, particles: list[Particle], particle_index: int) -> Particle:
        if self.normalization is None:
            raise RuntimeError("Normalization must be initialized before selecting lbest.")
        particle = particles[particle_index]
        candidates = [particles[i] for i in particle.neighborhood]
        return min(
            candidates,
            key=lambda candidate: (
                not candidate.pbest_evaluation.is_feasible,
                candidate.pbest_evaluation.total_violation,
                tchebycheff_scalarization(
                    candidate.pbest_evaluation.objectives,
                    particle.weight,
                    self.normalization,
                ),
            ),
        )

    def _step_particle(
        self,
        particle: Particle,
        lbest: Particle,
        world: WorldConfig,
        config: TrajectoryConfig,
        evaluator: TrajectoryEvaluator,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ) -> None:
        if self.normalization is None:
            raise RuntimeError("Normalization must be initialized before stepping particles.")

        velocity_limit = self._velocity_bounds(lower_bounds, upper_bounds)
        r1 = self.rng.random(size=particle.position.shape)
        r2 = self.rng.random(size=particle.position.shape)
        particle.velocity = (
            self.inertia * particle.velocity
            + self.cognitive * r1 * (particle.pbest_position - particle.position)
            + self.social * r2 * (lbest.pbest_position - particle.position)
        )
        particle.velocity = np.clip(particle.velocity, -velocity_limit, velocity_limit)
        particle.position = np.clip(particle.position + particle.velocity, lower_bounds, upper_bounds)

        trajectory = repair_trajectory(genes_to_trajectory(particle.position, config), world)
        particle.position = trajectory.genes.copy()
        particle.trajectory = trajectory
        particle.evaluation = self._evaluate(trajectory, evaluator)

        if evaluation_better_for_weight(
            particle.evaluation,
            particle.pbest_evaluation,
            particle.weight,
            self.normalization,
        ):
            particle.pbest_position = particle.position.copy()
            particle.pbest_evaluation = particle.evaluation

        self.archive.update(particle.trajectory, particle.evaluation, particle.position)

    def optimize(self, world: WorldConfig, config: TrajectoryConfig) -> list[Particle]:
        evaluator = TrajectoryEvaluator(world)
        lower_bounds, upper_bounds = self._get_bounds(config, world)
        particles = self._initialize_particles(world, config, evaluator, lower_bounds, upper_bounds)
        if not particles:
            return particles

        self._record_history(particles)
        while self.evals < self.max_evaluations:
            self.iteration += 1
            for i, particle in enumerate(particles):
                if self.evals >= self.max_evaluations:
                    break
                lbest = self._select_lbest(particles, i)
                self._step_particle(
                    particle,
                    lbest,
                    world,
                    config,
                    evaluator,
                    lower_bounds,
                    upper_bounds,
                )
            self._record_history(particles)

        return particles

    def _record_history(self, particles: list[Particle]) -> None:
        feasible = [particle for particle in particles if particle.evaluation.is_feasible]
        best_scalarized = None
        if self.normalization is not None and particles:
            values = [
                tchebycheff_scalarization(p.pbest_evaluation.objectives, p.weight, self.normalization)
                for p in particles
                if p.pbest_evaluation.is_feasible
            ]
            if values:
                best_scalarized = float(np.min(values))

        archive_objectives = np.array([entry.evaluation.objectives for entry in self.archive.entries])
        self.history.append(
            {
                "iteration": self.iteration,
                "evaluations": self.evals,
                "n_feasible": len(feasible),
                "archive_size": len(self.archive.entries),
                "best_scalarized": best_scalarized,
                "archive_objectives": archive_objectives,
            }
        )
