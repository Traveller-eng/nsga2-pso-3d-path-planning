from __future__ import annotations

import dataclasses
import numpy as np

from planner.environment.world import WorldConfig
from planner.representation.trajectory import TrajectoryConfig, Trajectory, create_straight_line_trajectory, create_perturbed_trajectory, repair_trajectory, genes_to_trajectory
from planner.evaluation.evaluator import TrajectoryEvaluator, EvaluationResult

@dataclasses.dataclass
class Individual:
    """
    Wraps a trajectory and its evaluation metrics for NSGA-II.
    """
    trajectory: Trajectory
    evaluation: EvaluationResult
    rank: int = -1
    crowding_distance: float = 0.0

def dominates(ind1: Individual, ind2: Individual) -> bool:
    """
    Returns True if ind1 dominates ind2 according to constrained dominance rules.
    1. Feasible > Infeasible
    2. If both infeasible: lower total_violation wins
    3. If both feasible: Pareto dominance on objectives (all minimized)
    """
    f1 = ind1.evaluation.is_feasible
    f2 = ind2.evaluation.is_feasible
    
    if f1 and not f2:
        return True
    elif not f1 and f2:
        return False
    elif not f1 and not f2:
        # Both infeasible, lower violation wins
        if ind1.evaluation.total_violation < ind2.evaluation.total_violation:
            return True
        elif ind1.evaluation.total_violation > ind2.evaluation.total_violation:
            return False
        else:
            return False
    else:
        # Both feasible, check Pareto dominance
        obj1 = ind1.evaluation.objectives
        obj2 = ind2.evaluation.objectives
        
        # ind1 strictly better in at least one, and no worse in any
        better_in_any = False
        for i in range(len(obj1)):
            if obj1[i] > obj2[i]:
                return False
            if obj1[i] < obj2[i]:
                better_in_any = True
        return better_in_any

def fast_non_dominated_sort(population: list[Individual]) -> list[list[Individual]]:
    """
    Standard NSGA-II fast non-dominated sort.
    Returns a list of fronts, where each front is a list of Individuals.
    """
    fronts: list[list[Individual]] = [[]]
    domination_count = {id(ind): 0 for ind in population}
    dominated_lists = {id(ind): [] for ind in population}
    
    for p in population:
        for q in population:
            if dominates(p, q):
                dominated_lists[id(p)].append(q)
            elif dominates(q, p):
                domination_count[id(p)] += 1
                
        if domination_count[id(p)] == 0:
            p.rank = 0
            fronts[0].append(p)
            
    i = 0
    while True:
        next_front = []
        for p in fronts[i]:
            for q in dominated_lists[id(p)]:
                domination_count[id(q)] -= 1
                if domination_count[id(q)] == 0:
                    q.rank = i + 1
                    next_front.append(q)
        if len(next_front) == 0:
            break
        fronts.append(next_front)
        i += 1
            
    return fronts

def calculate_crowding_distance(front: list[Individual]) -> None:
    """
    Calculates and sets the crowding distance for each individual in the front.
    """
    l = len(front)
    if l == 0:
        return
    if l <= 2:
        for ind in front:
            ind.crowding_distance = float('inf')
        return
        
    for ind in front:
        ind.crowding_distance = 0.0
        
    # Number of objectives
    n_obj = len(front[0].evaluation.objectives)
    
    for m in range(n_obj):
        # Sort front by m-th objective
        front.sort(key=lambda ind: ind.evaluation.objectives[m])
        
        # Boundary points get infinity
        front[0].crowding_distance = float('inf')
        front[-1].crowding_distance = float('inf')
        
        f_min = front[0].evaluation.objectives[m]
        f_max = front[-1].evaluation.objectives[m]
        
        if f_max - f_min == 0:
            continue
            
        for i in range(1, l - 1):
            if front[i].crowding_distance != float('inf'):
                dist = front[i+1].evaluation.objectives[m] - front[i-1].evaluation.objectives[m]
                front[i].crowding_distance += dist / (f_max - f_min)

def binary_tournament(population: list[Individual], rng: np.random.Generator) -> Individual:
    """
    Binary tournament selection based on rank and crowding distance.
    """
    i1, i2 = rng.choice(len(population), size=2, replace=False)
    p1 = population[i1]
    p2 = population[i2]
    
    # 1. Lower rank wins
    if p1.rank < p2.rank:
        return p1
    elif p2.rank < p1.rank:
        return p2
        
    # 2. If same rank, higher crowding distance wins
    if p1.crowding_distance > p2.crowding_distance:
        return p1
    elif p2.crowding_distance > p1.crowding_distance:
        return p2
        
    # 3. Tie break
    return p1 if rng.random() < 0.5 else p2

def sbx_crossover(p1: np.ndarray, p2: np.ndarray, prob: float, eta: float, lower_bounds: np.ndarray, upper_bounds: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulated Binary Crossover (SBX).
    """
    c1 = p1.copy()
    c2 = p2.copy()
    
    if rng.random() > prob:
        return c1, c2
        
    for i in range(len(p1)):
        if rng.random() <= 0.5 and abs(p1[i] - p2[i]) > 1e-14:
            y1 = min(p1[i], p2[i])
            y2 = max(p1[i], p2[i])
            yl = lower_bounds[i]
            yu = upper_bounds[i]
            
            rand = rng.random()
            
            beta = 1.0 + (2.0 * (y1 - yl) / (y2 - y1))
            alpha = 2.0 - beta ** -(eta + 1.0)
            if rand <= (1.0 / alpha):
                betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
            else:
                betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
            c1[i] = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
            
            beta = 1.0 + (2.0 * (yu - y2) / (y2 - y1))
            alpha = 2.0 - beta ** -(eta + 1.0)
            if rand <= (1.0 / alpha):
                betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
            else:
                betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
            c2[i] = 0.5 * ((y1 + y2) + betaq * (y2 - y1))
            
            # Clip to bounds
            c1[i] = min(max(c1[i], yl), yu)
            c2[i] = min(max(c2[i], yl), yu)
            
            if rng.random() <= 0.5:
                c1[i], c2[i] = c2[i], c1[i]
                
    return c1, c2

def polynomial_mutation(p: np.ndarray, prob: float, eta: float, lower_bounds: np.ndarray, upper_bounds: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Polynomial mutation.
    """
    c = p.copy()
    for i in range(len(p)):
        if rng.random() <= prob:
            y = c[i]
            yl = lower_bounds[i]
            yu = upper_bounds[i]
            if yl == yu:
                continue
                
            delta1 = (y - yl) / (yu - yl)
            delta2 = (yu - y) / (yu - yl)
            
            rand = rng.random()
            mut_pow = 1.0 / (eta + 1.0)
            
            if rand <= 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
                deltaq = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
                deltaq = 1.0 - val ** mut_pow
                
            y = y + deltaq * (yu - yl)
            y = min(max(y, yl), yu)
            c[i] = y
            
    return c

class NSGA2Optimizer:
    def __init__(self, 
                 population_size: int = 100, 
                 max_evaluations: int = 10000,
                 crossover_prob: float = 0.9,
                 crossover_eta: float = 20.0,
                 mutation_prob: float = None,  # defaults to 1/n_genes
                 mutation_eta: float = 20.0,
                 seed: int = 42):
        self.population_size = population_size
        self.max_evaluations = max_evaluations
        self.crossover_prob = crossover_prob
        self.crossover_eta = crossover_eta
        self.mutation_prob = mutation_prob
        self.mutation_eta = mutation_eta
        self.rng = np.random.default_rng(seed)
        
        self.evals = 0
        self.generation = 0
        self.history = []
        
    def _evaluate_trajectory(self, traj: Trajectory, evaluator: TrajectoryEvaluator) -> EvaluationResult:
        self.evals += 1
        return evaluator.evaluate(traj)
        
    def _get_bounds(self, config: TrajectoryConfig, world: WorldConfig) -> tuple[np.ndarray, np.ndarray]:
        # Expand bounds for each position gene
        n_free = config.n_free_points
        dim = config.dim
        lower = np.tile(world.lower_bounds, n_free)
        upper = np.tile(world.upper_bounds, n_free)
        return lower, upper

    def optimize(self, world: WorldConfig, config: TrajectoryConfig) -> list[Individual]:
        evaluator = TrajectoryEvaluator(world)
        lower_bounds, upper_bounds = self._get_bounds(config, world)
        
        if self.mutation_prob is None:
            self.mutation_prob = 1.0 / config.n_genes if config.n_genes > 0 else 0.1
            
        population = []
        
        # Initialize population
        if self.max_evaluations <= 0:
            return population
            
        # 1 straight line
        t0 = create_straight_line_trajectory(config)
        res = self._evaluate_trajectory(t0, evaluator)
        population.append(Individual(t0, res))
        
        # Perturbed versions
        scale = np.linalg.norm(world.upper_bounds - world.lower_bounds) * 0.05
        while len(population) < self.population_size and self.evals < self.max_evaluations:
            t_pert = create_perturbed_trajectory(config, self.rng, scale=scale)
            t_pert = repair_trajectory(t_pert, world)
            res = self._evaluate_trajectory(t_pert, evaluator)
            population.append(Individual(t_pert, res))
            
        # Initial sorting
        fronts = fast_non_dominated_sort(population)
        for front in fronts:
            calculate_crowding_distance(front)
            
        self._record_history(population, fronts)
        
        # Main loop
        while self.evals < self.max_evaluations:
            self.generation += 1
            offspring = []
            
            # Generate offspring
            while len(offspring) < self.population_size and self.evals + len(offspring) < self.max_evaluations:
                p1 = binary_tournament(population, self.rng)
                p2 = binary_tournament(population, self.rng)
                
                c1_genes, c2_genes = sbx_crossover(
                    p1.trajectory.genes, p2.trajectory.genes, 
                    self.crossover_prob, self.crossover_eta, 
                    lower_bounds, upper_bounds, self.rng
                )
                
                c1_genes = polynomial_mutation(
                    c1_genes, self.mutation_prob, self.mutation_eta, 
                    lower_bounds, upper_bounds, self.rng
                )
                c2_genes = polynomial_mutation(
                    c2_genes, self.mutation_prob, self.mutation_eta, 
                    lower_bounds, upper_bounds, self.rng
                )
                
                t_c1 = genes_to_trajectory(c1_genes, config)
                t_c2 = genes_to_trajectory(c2_genes, config)
                
                t_c1 = repair_trajectory(t_c1, world)
                t_c2 = repair_trajectory(t_c2, world)
                
                offspring.append(t_c1)
                if len(offspring) < self.population_size and self.evals + len(offspring) < self.max_evaluations:
                    offspring.append(t_c2)
                    
            # Evaluate offspring
            offspring_pop = []
            for t in offspring:
                res = self._evaluate_trajectory(t, evaluator)
                offspring_pop.append(Individual(t, res))
                
            # Combine and sort
            combined = population + offspring_pop
            fronts = fast_non_dominated_sort(combined)
            
            # Elitist selection
            new_population = []
            for front in fronts:
                calculate_crowding_distance(front)
                if len(new_population) + len(front) <= self.population_size:
                    new_population.extend(front)
                else:
                    # Sort front by crowding distance (descending)
                    front.sort(key=lambda x: x.crowding_distance, reverse=True)
                    remaining = self.population_size - len(new_population)
                    new_population.extend(front[:remaining])
                    break
                    
            population = new_population
            self._record_history(population, fast_non_dominated_sort(population))
            
        return population
        
    def _record_history(self, population: list[Individual], fronts: list[list[Individual]]):
        feasible_inds = [ind for ind in population if ind.evaluation.is_feasible]
        best_obj = None
        if feasible_inds:
            all_objs = np.array([ind.evaluation.objectives for ind in feasible_inds])
            best_obj = np.min(all_objs, axis=0)
            
        self.history.append({
            'generation': self.generation,
            'evaluations': self.evals,
            'n_feasible': len(feasible_inds),
            'f1_size': len(fronts[0]) if fronts else 0,
            'best_objectives': best_obj
        })
