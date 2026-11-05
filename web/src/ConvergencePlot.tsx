import { useState } from 'react';
import type { OptimizationResult } from './types';

type ObjectiveKey = 'path_length' | 'clearance_risk' | 'minimum_snap';
const OBJECTIVES: Array<{ key: ObjectiveKey; label: string; index: number }> = [
  { key: 'path_length', label: 'Path length', index: 0 },
  { key: 'clearance_risk', label: 'Clearance risk', index: 1 },
  { key: 'minimum_snap', label: 'Minimum snap', index: 2 },
];

export default function ConvergencePlot({ result }: { result: OptimizationResult | null }) {
  const [objective, setObjective] = useState<ObjectiveKey>('path_length');
  const selected = OBJECTIVES.find((item) => item.key === objective)!;
  const history = result?.evaluation_history ?? [];
  let best = Number.POSITIVE_INFINITY;
  const points = history.map((record) => {
    if (record.is_feasible) best = Math.min(best, record.objectives[selected.index]);
    return { evaluations: record.evaluations, value: Number.isFinite(best) ? best : null };
  }).filter((point): point is { evaluations: number; value: number } => point.value !== null);

  return (
    <section className="convergence-panel">
      <div className="instrument-toolbar">
        <span className="eyebrow">LIVE EVALUATION HISTORY</span>
        <div className="mini-select-group" role="group" aria-label="Convergence objective">
          {OBJECTIVES.map((item) => (
            <button key={item.key} className={objective === item.key ? 'mini-select active' : 'mini-select'} onClick={() => setObjective(item.key)}>{item.label}</button>
          ))}
        </div>
      </div>
      {!result ? (
        <div className="instrument-empty">Convergence appears after a backend optimization run.</div>
      ) : points.length < 2 ? (
        <div className="instrument-empty">No feasible evaluations were recorded for this run.</div>
      ) : (
        <div className="convergence-chart">
          <svg viewBox="0 0 700 172" preserveAspectRatio="none" role="img" aria-label={`${selected.label} best feasible value by objective evaluation`}>
            <path d="M44 18 V140 H684" className="chart-axis" />
            {[0, 1, 2, 3].map((line) => <path key={line} d={`M44 ${42 + line * 29} H684`} className="chart-grid" />)}
            <text x="44" y="160" className="chart-tick">0</text>
            <text x="680" y="160" className="chart-tick" textAnchor="end">{result.objective_evaluations} evaluations</text>
            <text x="13" y="23" className="chart-tick">{Math.min(...points.map((point) => point.value)).toPrecision(3)}</text>
            <text x="13" y="140" className="chart-tick">{Math.max(...points.map((point) => point.value)).toPrecision(3)}</text>
            <polyline
              points={points.map((point) => {
                const minValue = Math.min(...points.map((item) => item.value));
                const maxValue = Math.max(...points.map((item) => item.value));
                const x = 44 + (point.evaluations / result.objective_evaluations) * 640;
                const y = maxValue - minValue < 1e-10 ? 81 : 24 + ((point.value - minValue) / (maxValue - minValue)) * 112;
                return `${x},${y}`;
              }).join(' ')}
              className="convergence-line"
            />
          </svg>
          <span className="chart-caption">Best feasible objective recorded to evaluation budget · {selected.label}</span>
        </div>
      )}
    </section>
  );
}
