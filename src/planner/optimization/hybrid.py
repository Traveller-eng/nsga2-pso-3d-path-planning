from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from planner.environment.world import WorldConfig
from planner.evaluation.evaluator import EvaluationResult, TrajectoryEvaluator
from planner.optimization.mopso import (
    ObjectiveNormalization,
    Particle,
    build_neighborhoods,
    evaluation_better_for_weight,
    generate_weight_vectors,
    tchebycheff_scalarization,
)
from planner.optimization.nsga2 import (
    Individual,
    calculate_crowding_distance,
    fast_non_dominated_sort,
    polynomial_mutation,
    sbx_crossover,
    binary_tournament,
)
from planner.representation.trajectory import (
    Trajectory,
    TrajectoryConfig,
    create_perturbed_trajectory,
    create_straight_line_trajectory,
    genes_to_trajectory,
    repair_trajectory,
)


class HybridNSGA2PSOOptimizer:
    """NSGA-II with a Lamarckian PSO refinement step applied to selected elites."""

    def __init__(
        self,
        population_size: int = 100,
        max_evaluations: int = 10000,
        elite_fraction: float = 0.10,
        min_elites: int = 10,
        pso_iterations: int = 10,
        pso_inertia: float = 0.5,
        pso_cognitive: float = 1.5,
        pso_social: float = 1.5,
        velocity_clamp: float = 0.2,
        neighborhood_size: int = 10,
        writeback_probability: float = 1.0,
        seed: int = 42,
        crossover_prob: float = 0.9,
        crossover_eta: float = 20.0,
        mutation_prob: float | None = None,
        mutation_eta: float = 20.0,
    ):
        self.population_size = population_size
        self.max_evaluations = max_evaluations
        self.elite_fraction = elite_fraction
        self.min_elites = max(1, min_elites)
        self.pso_iterations = pso_iterations
        self.pso_inertia = pso_inertia
        self.pso_cognitive = pso_cognitive
        self.pso_social = pso_social
        self.velocity_clamp = velocity_clamp
        self.neighborhood_size = neighborhood_size
        self.writeback_probability = writeback_probability
        self.seed = seed
        self.crossover_prob = crossover_prob
        self.crossover_eta = crossover_eta
        self.mutation_prob = mutation_prob
        self.mutation_eta = mutation_eta

        self.rng = np.random.default_rng(seed)
        self.evals = 0
        self.nsga2_evaluations = 0
        self.pso_evaluations = 0
        self.generation = 0
        self.history: list[dict] = []

    def _get_bounds(self, config: TrajectoryConfig, world: WorldConfig) -> tuple[np.ndarray, np.ndarray]:
        lower = np.tile(world.lower_bounds, config.n_free_points)
        upper = np.tile(world.upper_bounds, config.n_free_points)
        return lower, upper

    def _velocity_bounds(self, lower_bounds: np.ndarray, upper_bounds: np.ndarray) -> np.ndarray:
        return self.velocity_clamp * (upper_bounds - lower_bounds)

    def _evaluate_nsga2(self, trajectory: Trajectory, evaluator: TrajectoryEvaluator) -> EvaluationResult:
        self.evals += 1
        self.nsga2_evaluations += 1
        return evaluator.evaluate(trajectory)

    def _evaluate_pso(self, trajectory: Trajectory, evaluator: TrajectoryEvaluator) -> EvaluationResult:
        self.evals += 1
        self.pso_evaluations += 1
        return evaluator.evaluate(trajectory)

    def _record_history(self, population: list[Individual], fronts: list[list[Individual]]) -> None:
        feasible = [ind for ind in population if ind.evaluation.is_feasible]
        best_obj = None
        if feasible:
            best_obj = np.min(np.array([ind.evaluation.objectives for ind in feasible]), axis=0)

        self.history.append(
            {
                "generation": self.generation,
                "evaluations": self.evals,
                "n_feasible": len(feasible),
                "first_front_size": len(fronts[0]) if fronts else 0,
                "elite_count": max(self.min_elites, int(math.ceil(self.elite_fraction * len(population)))),
                "pso_evaluations": self.pso_evaluations,
                "nsga2_evaluations": self.nsga2_evaluations,
                "best_objectives": best_obj,
            }
        )

    def _select_elites(self, population: list[Individual], elite_count: int | None = None) -> list[Individual]:
        if not population:
            return []
        if elite_count is None:
            elite_count = max(self.min_elites, int(math.ceil(self.elite_fraction * len(population))))
        elite_count = min(elite_count, len(population))

        fronts = fast_non_dominated_sort(population)
        for front in fronts:
            calculate_crowding_distance(front)

        ranked: list[Individual] = sorted(
            population,
            key=lambda ind: (
                not ind.evaluation.is_feasible,
                ind.rank,
                -float(ind.crowding_distance),
                ind.evaluation.total_violation,
            ),
        )

        selected: list[Individual] = []
        seen: set[int] = set()
        for ind in ranked:
            if len(selected) >= elite_count:
                break
            obj_id = id(ind)
            if obj_id in seen:
                continue
            selected.append(ind)
            seen.add(obj_id)
        return selected

    def _initialize_pso_particles(
        self,
        elites: list[Individual],
        world: WorldConfig,
        config: TrajectoryConfig,
        max_particles: int | None = None,
    ) -> list[Particle]:
        if not elites:
            return []
        if max_particles is None:
            max_particles = len(elites)
        max_particles = min(max_particles, len(elites))

        wrapper_elites = elites[:max_particles]
        weights = generate_weight_vectors(len(wrapper_elites), 3)
        if len(wrapper_elites) == 1:
            weights = np.array([[1.0, 0.0, 0.0]], dtype=float)

        lower_bounds, upper_bounds = self._get_bounds(config, world)
        velocity_limit = self._velocity_bounds(lower_bounds, upper_bounds)
        particles: list[Particle] = []
        for i, elite in enumerate(wrapper_elites):
            position = elite.trajectory.genes.copy()
            velocity = np.zeros_like(position) if position.size else np.array([], dtype=float)
            particle = Particle(
                position=position,
                velocity=velocity,
                trajectory=elite.trajectory,
                evaluation=elite.evaluation,
                pbest_position=position.copy(),
                pbest_evaluation=elite.evaluation,
                weight=weights[i].copy(),
                neighborhood=[],
            )
            particles.append(particle)

        neighborhoods = build_neighborhoods(np.array([p.weight for p in particles]), self.neighborhood_size)
        for particle, neighborhood in zip(particles, neighborhoods):
            particle.neighborhood = neighborhood

        return particles

    def _select_lbest(self, particles: list[Particle], particle_index: int, normalization: ObjectiveNormalization) -> Particle:
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
                    normalization,
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
        normalization: ObjectiveNormalization,
    ) -> None:
        velocity_limit = self._velocity_bounds(lower_bounds, upper_bounds)
        r1 = self.rng.random(size=particle.position.shape)
        r2 = self.rng.random(size=particle.position.shape)

        particle.velocity = (
            self.pso_inertia * particle.velocity
            + self.pso_cognitive * r1 * (particle.pbest_position - particle.position)
            + self.pso_social * r2 * (lbest.pbest_position - particle.position)
        )
        particle.velocity = np.clip(particle.velocity, -velocity_limit, velocity_limit)
        particle.position = np.clip(particle.position + particle.velocity, lower_bounds, upper_bounds)

        trajectory = repair_trajectory(genes_to_trajectory(particle.position, config), world)
        particle.position = trajectory.genes.copy()
        particle.trajectory = trajectory
        particle.evaluation = self._evaluate_pso(trajectory, evaluator)

        if evaluation_better_for_weight(
            particle.evaluation,
            particle.pbest_evaluation,
            particle.weight,
            normalization,
        ):
            particle.pbest_position = particle.position.copy()
            particle.pbest_evaluation = particle.evaluation

    def _run_pso_refinement(
        self,
        elites: list[Individual],
        world: WorldConfig,
        config: TrajectoryConfig,
        evaluator: TrajectoryEvaluator,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ) -> list[Particle]:
        particles = self._initialize_pso_particles(elites, world, config, len(elites))
        if not particles:
            return []

        objectives = np.array([particle.evaluation.objectives for particle in particles])
        normalization = ObjectiveNormalization.from_objectives(objectives)

        for _ in range(self.pso_iterations):
            if self.evals >= self.max_evaluations:
                break
            for i, particle in enumerate(particles):
                if self.evals >= self.max_evaluations:
                    break
                lbest = self._select_lbest(particles, i, normalization)
                self._step_particle(
                    particle,
                    lbest,
                    world,
                    config,
                    evaluator,
                    lower_bounds,
                    upper_bounds,
                    normalization,
                )
        return particles

    def _apply_lamarckian_writeback(
        self,
        population: list[Individual],
        elite_indices: list[int],
        refined_particles: list[Particle],
        writeback_probability: float | None = None,
    ) -> None:
        if writeback_probability is None:
            writeback_probability = self.writeback_probability

        for idx, particle in zip(elite_indices, refined_particles):
            if idx >= len(population):
                continue
            if writeback_probability >= 1.0 or self.rng.random() < writeback_probability:
                population[idx] = Individual(particle.trajectory, particle.evaluation)

    def optimize(self, world: WorldConfig, config: TrajectoryConfig) -> list[Individual]:
        evaluator = TrajectoryEvaluator(world)
        lower_bounds, upper_bounds = self._get_bounds(config, world)

        if self.mutation_prob is None:
            self.mutation_prob = 1.0 / config.n_genes if config.n_genes > 0 else 0.1

        population: list[Individual] = []
        if self.max_evaluations <= 0:
            return population

        initial = create_straight_line_trajectory(config)
        population.append(Individual(initial, self._evaluate_nsga2(initial, evaluator)))

        scale = float(np.linalg.norm(world.upper_bounds - world.lower_bounds) * 0.05)
        while len(population) < self.population_size and self.evals < self.max_evaluations:
            perturbed = create_perturbed_trajectory(config, self.rng, scale=scale)
            perturbed = repair_trajectory(perturbed, world)
            population.append(Individual(perturbed, self._evaluate_nsga2(perturbed, evaluator)))

        fronts = fast_non_dominated_sort(population)
        for front in fronts:
            calculate_crowding_distance(front)
        self._record_history(population, fronts)

        while self.evals < self.max_evaluations:
            self.generation += 1

            elite_count = max(self.min_elites, int(math.ceil(self.elite_fraction * self.population_size)))
            elite_count = min(elite_count, len(population))
            elites = self._select_elites(population, elite_count)
            elite_ids = {id(ind) for ind in elites}
            elite_indices = [i for i, member in enumerate(population) if id(member) in elite_ids]

            if elites and self.pso_iterations > 0 and self.evals < self.max_evaluations:
                refined_particles = self._run_pso_refinement(elites, world, config, evaluator, lower_bounds, upper_bounds)
                self._apply_lamarckian_writeback(population, elite_indices, refined_particles)

            fronts = fast_non_dominated_sort(population)
            for front in fronts:
                calculate_crowding_distance(front)

            offspring: list[Trajectory] = []
            while len(offspring) < self.population_size and self.evals < self.max_evaluations:
                p1 = binary_tournament(population, self.rng)
                p2 = binary_tournament(population, self.rng)

                c1_genes, c2_genes = sbx_crossover(
                    p1.trajectory.genes,
                    p2.trajectory.genes,
                    self.crossover_prob,
                    self.crossover_eta,
                    lower_bounds,
                    upper_bounds,
                    self.rng,
                )

                c1_genes = polynomial_mutation(
                    c1_genes,
                    self.mutation_prob,
                    self.mutation_eta,
                    lower_bounds,
                    upper_bounds,
                    self.rng,
                )
                c2_genes = polynomial_mutation(
                    c2_genes,
                    self.mutation_prob,
                    self.mutation_eta,
                    lower_bounds,
                    upper_bounds,
                    self.rng,
                )

                t_c1 = repair_trajectory(genes_to_trajectory(c1_genes, config), world)
                t_c2 = repair_trajectory(genes_to_trajectory(c2_genes, config), world)

                offspring.append(t_c1)
                if len(offspring) < self.population_size and self.evals < self.max_evaluations:
                    offspring.append(t_c2)

            offspring_pop: list[Individual] = []
            for traj in offspring:
                offspring_pop.append(Individual(traj, self._evaluate_nsga2(traj, evaluator)))

            combined = population + offspring_pop
            fronts = fast_non_dominated_sort(combined)
            for front in fronts:
                calculate_crowding_distance(front)

            new_population: list[Individual] = []
            for front in fronts:
                if len(new_population) + len(front) <= self.population_size:
                    new_population.extend(front)
                else:
                    front_sorted = sorted(front, key=lambda x: x.crowding_distance, reverse=True)
                    remaining = self.population_size - len(new_population)
                    new_population.extend(front_sorted[:remaining])
                    break

            population = new_population
            fronts = fast_non_dominated_sort(population)
            for front in fronts:
                calculate_crowding_distance(front)
            self._record_history(population, fronts)

            if len(population) == 0:
                break

        return population
