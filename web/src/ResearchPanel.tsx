import { Activity, ArrowDownRight, ArrowUpRight, Database, FlaskConical } from 'lucide-react';
import type { ResearchData } from './types';

function format(value: number | null | undefined, digits = 3): string {
  return value == null || !Number.isFinite(value) ? '—' : value.toFixed(digits);
}

export default function ResearchPanel({ research, error }: { research: ResearchData | null; error: string | null }) {
  if (error) return <div className="instrument-empty research-error">Research artifacts unavailable: {error}</div>;
  if (!research) return <div className="instrument-empty">Loading committed research artifacts…</div>;

  const dynamic = research.dynamic_replanning;
  const finalEvent = String(dynamic.event_count - 1);
  const final = dynamic.events[finalEvent];
  const coldRate = final?.conditions.cold.feasibility_rate?.mean ?? null;
  const warmRate = final?.conditions.warm.feasibility_rate?.mean ?? null;
  const coldRuntime = final?.conditions.cold.wall_clock_seconds?.mean ?? null;
  const warmRuntime = final?.conditions.warm.wall_clock_seconds?.mean ?? null;
  const coldHv = final?.conditions.cold.hypervolume?.mean ?? null;
  const warmHv = final?.conditions.warm.hypervolume?.mean ?? null;
  const paired = (metric: string) => dynamic.paired_differences.find((row) => row.event_index === dynamic.event_count - 1 && row.metric === metric);
  const runtimeDiff = paired('wall_clock_seconds');
  const hvDiff = paired('hypervolume');
  const feasibilityDiff = paired('feasibility_rate');

  return (
    <div className="research-content">
      <div className="research-heading">
        <div><span className="eyebrow"><Database size={12} /> STORED RESEARCH</span><h3>Evidence, not animation</h3></div>
        <span className="offline-stamp"><span /> FROZEN ARTIFACTS</span>
      </div>
      <div className="research-block">
        <div className="research-block-heading"><FlaskConical size={15} /><span>Dynamic replanning · event {dynamic.event_count - 1}</span><small>{dynamic.paired_seed_count} paired seeds · {dynamic.budget_per_event} evaluations / event</small></div>
        <div className="research-metrics">
          <ResearchMetric label="Feasible population rate" cold={coldRate} warm={warmRate} formatValue={(value) => `${format((value ?? 0) * 100, 1)}%`} difference={feasibilityDiff?.mean ?? null} higherIsBetter />
          <ResearchMetric label="Replanning latency" cold={coldRuntime} warm={warmRuntime} formatValue={(value) => `${format(value, 3)} s`} difference={runtimeDiff?.mean ?? null} higherIsBetter={false} />
          <ResearchMetric label="Hypervolume · scaled" cold={coldHv} warm={warmHv} formatValue={(value) => format(value, 3)} difference={hvDiff?.mean ?? null} higherIsBetter />
        </div>
        <p className="research-caveat">{dynamic.interpretation}</p>
        <details className="research-method">
          <summary>Statistical readout</summary>
          <div className="method-grid">
            <span>Latency · mean Δ</span><b>{format(runtimeDiff?.mean, 4)} s</b>
            <span>Latency · adjusted tests</span><b>t {format(runtimeDiff?.paired_t_test_p_holm, 3)} · W {format(runtimeDiff?.wilcoxon_p_holm, 3)}</b>
            <span>HV · adjusted tests</span><b>t {format(hvDiff?.paired_t_test_p_holm, 3)} · W {format(hvDiff?.wilcoxon_p_holm, 3)}</b>
            <span>Feasibility · mean Δ</span><b>{format((feasibilityDiff?.mean ?? 0) * 100, 1)} pp</b>
          </div>
          <p>Warm-start minus cold-start. Median and confidence intervals are in the saved analysis artifact. Missing feasible-only outcomes remain undefined rather than being imputed.</p>
        </details>
      </div>
      <div className="research-block benchmark-block">
        <div className="research-block-heading"><Activity size={15} /><span>Static algorithm benchmark</span><small>Offline · {research.static_benchmark.objective_space} objective space</small></div>
        <div className="benchmark-list">
          {research.static_benchmark.algorithms.map((algorithm) => (
            <div className="benchmark-row" key={algorithm.algorithm}>
              <b>{algorithm.algorithm}</b>
              <span>{algorithm.seed_count} seeds · {algorithm.evaluation_budget} evaluations</span>
              <span>HV {format(algorithm.mean_hypervolume, 3)}</span>
              <span>IGD {format(algorithm.mean_igd, 3)}</span>
            </div>
          ))}
        </div>
        <p className="research-caveat">This is a different static, obstacle-free scenario and uses a raw objective-space reference front. Do not compare its raw HV/IGD values with the normalized dynamic results above.</p>
      </div>
    </div>
  );
}

function ResearchMetric({
  label,
  cold,
  warm,
  formatValue,
  difference,
  higherIsBetter,
}: {
  label: string;
  cold: number | null;
  warm: number | null;
  formatValue: (value: number | null) => string;
  difference: number | null;
  higherIsBetter: boolean;
}) {
  const favorable = difference != null && (higherIsBetter ? difference > 0 : difference < 0);
  const unfavorable = difference != null && (higherIsBetter ? difference < 0 : difference > 0);
  return (
    <div className="research-metric">
      <span>{label}</span>
      <div><small>COLD</small><b>{formatValue(cold)}</b></div>
      <div><small>WARM</small><b>{formatValue(warm)}</b></div>
      <span className={`metric-delta ${favorable ? 'favorable' : ''} ${unfavorable ? 'unfavorable' : ''}`}>
        {difference == null ? 'Δ —' : <>{difference > 0 ? <ArrowUpRight size={12} /> : difference < 0 ? <ArrowDownRight size={12} /> : null}{format(difference, 3)} Δ</>}
      </span>
    </div>
  );
}
