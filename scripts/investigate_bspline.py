import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from planner.geometry.bspline import create_trajectory_bspline, evaluate, evaluate_derivative
from planner.geometry.sampling import parametric_to_physical

def investigate():
    n_cp = 8
    degree = 5
    t_max = 5.0
    
    # 1. Construct collinear, evenly spaced control points from (0,0,0) to (10,10,10)
    start = np.array([0.0, 0.0, 0.0])
    goal = np.array([10.0, 10.0, 10.0])
    fractions = np.linspace(0, 1, n_cp).reshape(-1, 1)
    cps = start + fractions * (goal - start)
    
    # Create curve
    curve = create_trajectory_bspline(cps, degree=degree)
    
    print("--- KNOT VECTOR ---")
    print(curve.knot_vector)
    
    # 2. Evaluate at t=0 and t=1 (parametric)
    # Using small t values for numerical stability if exactly 0 or 1 has issues (it shouldn't)
    u_eval = np.array([0.0, 1.0])
    
    p = evaluate(curve, u_eval)
    v_param = evaluate_derivative(curve, u_eval, 1)
    a_param = evaluate_derivative(curve, u_eval, 2)
    s_param = evaluate_derivative(curve, u_eval, 4)
    
    # Convert to physical
    samples = {
        'u': u_eval,
        'positions': p,
        'param_vel': v_param,
        'param_acc': a_param,
        'param_snap': s_param
    }
    phys = parametric_to_physical(samples, t_max)
    
    print("\n--- ENDPOINT BEHAVIOR ---")
    print(f"p(0) = {phys['positions'][0]}")
    print(f"p(T) = {phys['positions'][1]}")
    
    print(f"\nv(0) = {phys['velocity'][0]}")
    print(f"v(T) = {phys['velocity'][1]}")
    
    print(f"\na(0) = {phys['acceleration'][0]}")
    print(f"a(T) = {phys['acceleration'][1]}")
    
    print(f"\nsnap(0) = {phys['snap'][0]}")
    print(f"snap(T) = {phys['snap'][1]}")
    
    print("\n--- BEHAVIOR ANALYSIS ---")
    p0_match = np.allclose(phys['positions'][0], start)
    pT_match = np.allclose(phys['positions'][1], goal)
    v0_zero = np.allclose(phys['velocity'][0], 0.0)
    vT_zero = np.allclose(phys['velocity'][1], 0.0)
    a0_zero = np.allclose(phys['acceleration'][0], 0.0)
    aT_zero = np.allclose(phys['acceleration'][1], 0.0)
    
    print(f"p(0) == start: {p0_match}")
    print(f"p(T) == goal:  {pT_match}")
    print(f"v(0) == 0:     {v0_zero}")
    print(f"v(T) == 0:     {vT_zero}")
    print(f"a(0) == 0:     {a0_zero}")
    print(f"a(T) == 0:     {aT_zero}")
    
    # Look at speed at different points
    u_dense = np.linspace(0, 1, 100)
    v_dense = evaluate_derivative(curve, u_dense, 1) / t_max
    speeds = np.linalg.norm(v_dense, axis=1)
    print(f"\nMax speed: {np.max(speeds):.2f}, Min speed: {np.min(speeds):.2f}")
    
if __name__ == '__main__':
    investigate()
