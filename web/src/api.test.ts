import { afterEach, describe, expect, it, vi } from 'vitest';
import { createOptimizationJob } from './api';

describe('createOptimizationJob', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('sends the optimization request and returns the created job', async () => {
    const payload = {
      algorithm: 'nsga2' as const,
      scenario_id: 'dynamic' as const,
      event_index: 0,
      start: [0, 0, 0] as [number, number, number],
      goal: [9, 9, 9] as [number, number, number],
      population_size: 20,
      max_evaluations: 300,
      seed: 42,
    };
    const responseBody = { job_id: 'job-1', status: 'running' as const, algorithm: 'nsga2' as const };
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(responseBody), { status: 201 }),
    );
    vi.stubGlobal('fetch', fetchMock);

    await expect(createOptimizationJob(payload)).resolves.toEqual(responseBody);
    expect(fetchMock).toHaveBeenCalledWith('/api/jobs', {
      method: 'POST',
      body: JSON.stringify(payload),
      headers: { 'Content-Type': 'application/json' },
    });
  });

  it('surfaces the API error message for failed requests', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ error: 'invalid optimization request' }), { status: 400 }),
    ));

    await expect(createOptimizationJob({
      algorithm: 'nsga2',
      scenario_id: 'dynamic',
      event_index: 0,
      start: [0, 0, 0],
      goal: [9, 9, 9],
      population_size: 20,
      max_evaluations: 300,
      seed: 42,
    })).rejects.toThrow('invalid optimization request');
  });
});
