import type { AlgorithmId, JobState, ResearchData, ScenarioData, ScenarioId, Vec3 } from './types';

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.error ?? `Request failed (${response.status})`);
  }
  return body as T;
}

export function getScenario(scenario: ScenarioId, event: number, start: Vec3, goal: Vec3): Promise<ScenarioData> {
  const query = new URLSearchParams({ scenario, event: String(event), start: start.join(','), goal: goal.join(',') });
  return request(`/api/scenario?${query.toString()}`);
}

export function getResearch(): Promise<ResearchData> {
  return request('/api/research');
}

export function createOptimizationJob(payload: {
  algorithm: AlgorithmId;
  scenario_id: ScenarioId;
  event_index: number;
  start: Vec3;
  goal: Vec3;
  population_size: number;
  max_evaluations: number;
  seed: number;
  initial_population?: number[][];
}): Promise<{ job_id: string; status: 'running'; algorithm: AlgorithmId }> {
  return request('/api/jobs', { method: 'POST', body: JSON.stringify(payload) });
}

export function getJob(jobId: string): Promise<JobState> {
  return request(`/api/jobs/${jobId}`);
}

export function cancelJob(jobId: string): Promise<JobState> {
  return request(`/api/jobs/${jobId}`, { method: 'DELETE' });
}
