import { useState } from 'react';
import type { Candidate } from './types';

type ParetoPlotProps = {
  candidates: Candidate[];
  selectedIndex: number | null;
  onSelect: (index: number) => void;
};

export default function ParetoPlot({ candidates, selectedIndex, onSelect }: ParetoPlotProps) {
  const [hovered, setHovered] = useState<number | null>(null);
  const plot = candidates.filter((candidate) => candidate.rank === 0 || candidate.index === selectedIndex);
  if (!plot.length) {
    return <div className="instrument-empty">Run an optimizer to plot its current non-dominated candidates.</div>;
  }

  const values = plot.map((candidate) => [candidate.objectives.path_length, candidate.objectives.clearance_risk, candidate.objectives.minimum_snap]);
  const limits = [0, 1, 2].map((axis) => {
    const column = values.map((value) => value[axis]);
    return [Math.min(...column), Math.max(...column)] as const;
  });
  const normalize = (value: number, axis: number) => {
    const [minimum, maximum] = limits[axis];
    return maximum - minimum < 1e-9 ? 0.5 : (value - minimum) / (maximum - minimum);
  };
  const points = plot.map((candidate) => {
    const xValue = normalize(candidate.objectives.path_length, 0);
    const yValue = normalize(candidate.objectives.clearance_risk, 1);
    const zValue = normalize(candidate.objectives.minimum_snap, 2);
    return {
      candidate,
      x: 54 + xValue * 314 + zValue * 24,
      y: 170 - yValue * 124 - zValue * 22,
    };
  });
  const hoveredCandidate = candidates.find((candidate) => candidate.index === hovered);

  return (
    <div className="pareto-wrap">
      <svg className="pareto-svg" viewBox="0 0 430 214" role="img" aria-label="Three-objective Pareto trade-off projection">
        <path d="M54 171 L370 171 M54 171 L54 35 M54 171 L78 148" className="pareto-axis" />
        <path d="M54 137 L370 137 M54 103 L370 103 M54 69 L370 69" className="pareto-grid" />
        <path d="M54 171 L78 148 L394 148" className="pareto-axis pareto-depth" />
        <text x="210" y="201" className="pareto-axis-label">PATH LENGTH · m</text>
        <text x="12" y="102" className="pareto-axis-label" transform="rotate(-90 12 102)">CLEARANCE RISK</text>
        <text x="292" y="164" className="pareto-axis-label pareto-snap-label">MINIMUM SNAP ↗</text>
        {points.map(({ candidate, x, y }) => (
          <circle
            key={candidate.index}
            cx={x}
            cy={y}
            r={candidate.index === selectedIndex ? 6 : 4.2}
            className={[
              'pareto-point',
              candidate.is_feasible ? 'is-feasible' : 'is-infeasible',
              candidate.index === selectedIndex ? 'is-selected' : '',
            ].join(' ')}
            onClick={() => onSelect(candidate.index)}
            onMouseEnter={() => setHovered(candidate.index)}
            onMouseLeave={() => setHovered(null)}
            role="button"
            tabIndex={0}
            aria-label={`Candidate ${candidate.index + 1}, path ${candidate.objectives.path_length.toFixed(2)}, risk ${candidate.objectives.clearance_risk.toFixed(3)}, snap ${candidate.objectives.minimum_snap.toFixed(3)}`}
            onKeyDown={(event) => {
              if (event.key === 'Enter' || event.key === ' ') onSelect(candidate.index);
            }}
          />
        ))}
      </svg>
      <div className="pareto-readout" aria-live="polite">
        {hoveredCandidate ? (
          <><span>CANDIDATE {String(hoveredCandidate.index + 1).padStart(2, '0')}</span><b>{hoveredCandidate.objectives.path_length.toFixed(2)} m</b><b>{hoveredCandidate.objectives.clearance_risk.toFixed(3)} risk</b><b>{hoveredCandidate.objectives.minimum_snap.toFixed(3)} snap</b></>
        ) : (
          <><span>SELECT A POINT</span><span>Trade-offs across three minimized objectives</span></>
        )}
      </div>
    </div>
  );
}
