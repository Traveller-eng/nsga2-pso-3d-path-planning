export type Vec3 = [number, number, number];
export type AlgorithmId = 'nsga2' | 'mopso' | 'hybrid';
export type ScenarioId = 'static' | 'dynamic';

export type Obstacle = {
  kind: 'sphere' | 'aabb';
  center?: Vec3;
  radius?: number;
  lower?: Vec3;
  upper?: Vec3;
  motion?:
    | { kind: 'piecewise_linear'; path: Vec3[]; path_times: number[] }
    | { kind: 'linear'; initial_center: Vec3; velocity: Vec3 }
    | { kind: 'linear_aabb'; initial_lower: Vec3; initial_upper: Vec3; velocity: Vec3 };
};

export type ScenarioData = {
  scenario_id: ScenarioId;
  event_index: number;
  event_time: number;
  event_times: number[];
  world: {
    bounds_min: Vec3;
    bounds_max: Vec3;
    start: Vec3;
    goal: Vec3;
    t_max: number;
    v_max: number;
    a_max: number;
    d_safe: number;
    d_comfort: number;
    obstacles: Obstacle[];
  };
  trajectory: { n_control_points: number; degree: number; dim: number };
};

export type Candidate = {
  index: number;
  rank: number;
  genes: number[];
  control_points: Vec3[];
  positions: Vec3[];
  trajectory_times: number[];
  objectives: { path_length: number; clearance_risk: number; minimum_snap: number };
  is_feasible: boolean;
  total_violation: number;
  violation_details: Record<string, number>;
  minimum_clearance: number;
  max_speed: number;
  max_acceleration: number;
};

export type OptimizerHistory = Record<string, unknown>;

export type OptimizationResult = {
  status: 'complete';
  algorithm: AlgorithmId;
  scenario_id: ScenarioId;
  event_index: number;
  event_time: number;
  start: Vec3;
  goal: Vec3;
  seed: number;
  objective_evaluations: number;
  imported_trajectory_count: number;
  accepted_imported_count: number;
  reevaluated_imported_count: number;
  accepted_imported_feasible_count: number;
  optimization_seconds: number;
  wall_clock_seconds: number;
  generation_count: number;
  feasible_count: number;
  feasibility_rate: number;
  best_candidate_index: number | null;
  history: OptimizerHistory[];
  evaluation_history: Array<{ evaluations: number; objectives: Vec3; is_feasible: boolean }>;
  candidates: Candidate[];
};

export type JobState =
  | { job_id: string; status: 'running'; elapsed_seconds: number; algorithm: AlgorithmId; message: string }
  | { job_id: string; status: 'cancelling'; elapsed_seconds: number; algorithm: AlgorithmId; message: string }
  | { job_id: string; status: 'complete'; result: OptimizationResult }
  | { job_id: string; status: 'failed' | 'cancelled'; error: string };

export type ResearchData = {
  static_benchmark: {
    configuration: string;
    objective_space: string;
    scenario: string;
    common_reference_front_size: number;
    algorithms: Array<{
      algorithm: string;
      seed_count: number;
      evaluation_budget: number;
      mean_runtime_seconds: number;
      mean_hypervolume: number;
      mean_igd: number;
      mean_feasible_count: number;
    }>;
  };
  dynamic_replanning: {
    configuration: string;
    paired_seed_count: number;
    budget_per_event: number;
    event_count: number;
    events: Record<string, { event_time: number; conditions: Record<string, Record<string, { n: number; mean: number | null; median: number | null; std_sample: number | null; mean_ci95: number[] | null }>> }>;
    paired_differences: Array<{
      event_index: number;
      metric: string;
      n: number;
      mean: number | null;
      median: number | null;
      std_sample: number | null;
      mean_ci95: number[] | null;
      paired_t_test_p_holm: number | null;
      wilcoxon_p_holm: number | null;
      included_in_primary_test_family: boolean;
    }>;
    interpretation: string;
  };
};
