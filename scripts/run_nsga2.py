import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from planner.environment.world import make_3d_static_world
from planner.representation.trajectory import TrajectoryConfig
from planner.optimization.nsga2 import NSGA2Optimizer

def run():
    world = make_3d_static_world(
        bounds_max=(10.0, 10.0, 10.0),
        start=(0.0, 0.0, 0.0),
        goal=(10.0, 10.0, 10.0),
        obstacles=[]
    )
    # Generous limits for a clean environment so things are feasible
    world.v_max = 100.0
    world.a_max = 1000.0
    
    config = TrajectoryConfig(8, 3, world.start, world.goal, world.t_max)
    
    pop_size = 50
    max_evals = 1000
    opt = NSGA2Optimizer(population_size=pop_size, max_evaluations=max_evals, seed=42)
    
    start_time = time.time()
    final_pop = opt.optimize(world, config)
    end_time = time.time()
    
    feasible_inds = [ind for ind in final_pop if ind.evaluation.is_feasible]
    feasible_count = len(feasible_inds)
    
    if feasible_count > 0:
        objs = np.array([ind.evaluation.objectives for ind in feasible_inds])
        min_obj = np.min(objs, axis=0)
        max_obj = np.max(objs, axis=0)
    else:
        min_obj = max_obj = [float('nan')] * 3
        
    first_front = [ind for ind in final_pop if ind.rank == 0]
    
    print("--- NSGA-II Integration Experiment ---")
    print(f"Population Size      : {pop_size}")
    print(f"Evaluation Budget    : {max_evals}")
    print(f"Generations          : {opt.generation}")
    print(f"Feasible Count       : {feasible_count}")
    print(f"First Front Size     : {len(first_front)}")
    print(f"Objective Ranges:")
    print(f"  Path Length        : [{min_obj[0]:.4f}, {max_obj[0]:.4f}]")
    print(f"  Clearance Risk     : [{min_obj[1]:.4f}, {max_obj[1]:.4f}]")
    print(f"  Snap Obj (log)     : [{min_obj[2]:.4f}, {max_obj[2]:.4f}]")
    print(f"Runtime              : {end_time - start_time:.4f} seconds")

if __name__ == '__main__':
    run()
