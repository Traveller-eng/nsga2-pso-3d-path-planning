import {
  Activity,
  AlertTriangle,
  Aperture,
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  Check,
  ChevronDown,
  CircleStop,
  Compass,
  Crosshair,
  Gauge,
  Orbit,
  Pause,
  Play,
  RotateCcw,
  Settings2,
  ShieldAlert,
  SlidersHorizontal,
  Sparkles,
  Target,
  Timer,
  Waypoints,
  X,
} from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { cancelJob, createOptimizationJob, getJob, getResearch, getScenario } from './api';
import ConvergencePlot from './ConvergencePlot';
import ParetoPlot from './ParetoPlot';
import ResearchPanel from './ResearchPanel';
import WorldScene from './WorldScene';
import type { CameraPreset } from './WorldScene';
import type { AlgorithmId, Candidate, JobState, OptimizationResult, ResearchData, ScenarioData, ScenarioId, Vec3 } from './types';

const ALGORITHMS: Array<{ id: AlgorithmId; label: string; short: string }> = [
  { id: 'nsga2', label: 'NSGA-II', short: 'NSGA' },
  { id: 'mopso', label: 'MOPSO', short: 'MOPSO' },
  { id: 'hybrid', label: 'Hybrid · Lamarckian PSO', short: 'HYBRID' },
];

const CAMERA_PRESETS: Array<{ id: CameraPreset; label: string; icon: typeof Compass }> = [
  { id: 'overview', label: 'Overview', icon: Compass },
  { id: 'side', label: 'Side profile', icon: Aperture },
  { id: 'top', label: 'Top-down', icon: Crosshair },
  { id: 'goal', label: 'Goal inspection', icon: Target },
];

function timeLabel(value: number): string {
  const minutes = Math.floor(value / 60).toString().padStart(2, '0');
  const seconds = (value % 60).toFixed(2).padStart(5, '0');
  return `${minutes}:${seconds}`;
}

function objectiveLabel(value: number | undefined, digits = 3): string {
  return value == null || !Number.isFinite(value) ? '—' : value.toFixed(digits);
}

function hardFeasible(candidate: Candidate | null | undefined, scenario: ScenarioData | null): boolean {
  if (!candidate || !scenario) return false;
  const hardViolation = candidate.total_violation <= 1e-12
    && Object.values(candidate.violation_details).every((value) => Math.abs(value) <= 1e-12);
  const clearanceOk = candidate.minimum_clearance + 1e-12 >= scenario.world.d_safe;
  return candidate.is_feasible && hardViolation && clearanceOk;
}

export default function App() {
  const [scenarioId, setScenarioId] = useState<ScenarioId>('dynamic');
  const [eventIndex, setEventIndex] = useState(0);
  const [startPoint, setStartPoint] = useState<Vec3>([0, 0, 0]);
  const [goalPoint, setGoalPoint] = useState<Vec3>([9, 9, 9]);
  const [scenario, setScenario] = useState<ScenarioData | null>(null);
  const [scenarioError, setScenarioError] = useState<string | null>(null);
  const [research, setResearch] = useState<ResearchData | null>(null);
  const [researchError, setResearchError] = useState<string | null>(null);
  const [algorithm, setAlgorithm] = useState<AlgorithmId>('nsga2');
  const [populationSize, setPopulationSize] = useState(20);
  const [evaluationBudget, setEvaluationBudget] = useState(300);
  const [seed, setSeed] = useState(42);
  const [warmStart, setWarmStart] = useState(true);
  const [showObstacles, setShowObstacles] = useState(true);
  const [showCandidates, setShowCandidates] = useState(true);
  const [showControlPoints, setShowControlPoints] = useState(true);
  const [showReference, setShowReference] = useState(true);
  const [followVehicle, setFollowVehicle] = useState(false);
  const [quality, setQuality] = useState<'balanced' | 'high'>('balanced');
  const [cameraPreset, setCameraPreset] = useState<CameraPreset>('overview');
  const [currentTime, setCurrentTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [activeTab, setActiveTab] = useState<'pareto' | 'convergence' | 'research'>('pareto');
  const [drawerOpen, setDrawerOpen] = useState(true);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [confirmRun, setConfirmRun] = useState(false);
  const [currentResult, setCurrentResult] = useState<OptimizationResult | null>(null);
  const [selectedCandidateIndex, setSelectedCandidateIndex] = useState<number | null>(null);
  const [previousCandidate, setPreviousCandidate] = useState<Candidate | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobState, setJobState] = useState<JobState | null>(null);
  const [jobError, setJobError] = useState<string | null>(null);
  const [environmentHealth, setEnvironmentHealth] = useState<'loading' | 'ready' | 'error'>('loading');
  const [reducedMotion, setReducedMotion] = useState(() => window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  const playbackRef = useRef(0);

  const startKey = startPoint.join(',');
  const goalKey = goalPoint.join(',');
  const currentEvent = scenario?.event_times[eventIndex] ?? 0;
  const duration = scenario?.world.t_max ?? 0;
  const resultMatchesView = currentResult?.scenario_id === scenarioId
    && currentResult?.event_index === eventIndex
    && currentResult?.algorithm === algorithm
    && currentResult.start.every((value, axis) => Math.abs(value - startPoint[axis]) < 1e-9)
    && currentResult.goal.every((value, axis) => Math.abs(value - goalPoint[axis]) < 1e-9);
  const visibleResult = resultMatchesView ? currentResult : null;
  const candidates = visibleResult?.candidates ?? [];
  const selectedCandidate = candidates.find((candidate) => candidate.index === selectedCandidateIndex) ?? null;
  const canWarmStart = scenarioId === 'dynamic' && algorithm === 'nsga2' && currentResult?.scenario_id === 'dynamic' && currentResult.event_index !== eventIndex;
  const running = jobState?.status === 'running' || jobState?.status === 'cancelling';
  const cancelling = jobState?.status === 'cancelling';
  const selectedAlgorithm = ALGORITHMS.find((item) => item.id === algorithm)!;

  useEffect(() => {
    let current = true;
    setEnvironmentHealth('loading');
    setScenarioError(null);
    getScenario(scenarioId, eventIndex, startPoint, goalPoint).then((data) => {
      if (!current) return;
      setScenario(data);
      setEnvironmentHealth('ready');
      playbackRef.current = 0;
      setCurrentTime(0);
      setPlaying(false);
    }).catch((error: Error) => {
      if (!current) return;
      setScenarioError(error.message);
      setEnvironmentHealth('error');
    });
    return () => { current = false; };
  }, [scenarioId, eventIndex, startKey, goalKey]);

  useEffect(() => {
    let current = true;
    getResearch().then((data) => { if (current) setResearch(data); }).catch((error: Error) => { if (current) setResearchError(error.message); });
    return () => { current = false; };
  }, []);

  useEffect(() => {
    document.documentElement.classList.toggle('reduced-motion', reducedMotion);
  }, [reducedMotion]);

  useEffect(() => {
    if (!jobId) return;
    let current = true;
    let timer = 0;
    const poll = async () => {
      try {
        const next = await getJob(jobId);
        if (!current) return;
        setJobState(next);
        if (next.status === 'complete') {
          if (selectedCandidate) setPreviousCandidate(selectedCandidate);
          setCurrentResult(next.result);
          setSelectedCandidateIndex(next.result.best_candidate_index);
          setJobId(null);
        } else if (next.status === 'failed') {
          setJobError(next.error);
          setJobId(null);
        } else if (next.status === 'cancelled') {
          setJobId(null);
        }
      } catch (error) {
        if (!current) return;
        setJobError((error as Error).message);
        setJobId(null);
      }
    };
    void poll();
    timer = window.setInterval(() => { void poll(); }, 650);
    return () => { current = false; window.clearInterval(timer); };
  }, [jobId]);

  useEffect(() => {
    if (!playing || !selectedCandidate || duration <= 0) return;
    let frame = 0;
    let previousFrameTime = 0;
    let value = playbackRef.current;
    const advance = (timestamp: number) => {
      if (!previousFrameTime) previousFrameTime = timestamp;
      const delta = Math.min((timestamp - previousFrameTime) / 1000, 0.08) * playbackSpeed;
      previousFrameTime = timestamp;
      value = Math.min(duration, value + delta);
      playbackRef.current = value;
      setCurrentTime(value);
      if (value >= duration) {
        setPlaying(false);
        return;
      }
      frame = window.requestAnimationFrame(advance);
    };
    frame = window.requestAnimationFrame(advance);
    return () => window.cancelAnimationFrame(frame);
  }, [playing, selectedCandidateIndex, duration, playbackSpeed]);

  useEffect(() => {
    if (!canWarmStart) setWarmStart(false);
  }, [canWarmStart]);

  const feasibleCount = visibleResult?.feasible_count ?? null;
  const selectedFeasible = hardFeasible(selectedCandidate, scenario);
  const selectedClearanceMargin = selectedCandidate && scenario ? selectedCandidate.minimum_clearance - scenario.world.d_safe : undefined;
  const selectedState = selectedFeasible ? 'FEASIBLE' : selectedCandidate ? 'INFEASIBLE' : 'NO PLAN';
  const selectedStatusClass = selectedFeasible ? 'status-feasible' : selectedCandidate ? 'status-warning' : 'status-muted';
  const liveElapsed = jobState?.status === 'running' ? jobState.elapsed_seconds : 0;

  function setPlayback(value: number) {
    playbackRef.current = value;
    setCurrentTime(value);
  }

  async function beginOptimization() {
    if (!scenario) return;
    setConfirmRun(false);
    setJobError(null);
    setJobState({ job_id: 'starting', status: 'running', elapsed_seconds: 0, algorithm, message: 'Starting Python optimizer' });
    const initialPopulation = warmStart && canWarmStart
      ? currentResult?.candidates.slice(0, populationSize).map((candidate) => candidate.genes) ?? []
      : [];
    try {
      const job = await createOptimizationJob({
        algorithm,
        scenario_id: scenarioId,
        event_index: eventIndex,
        start: startPoint,
        goal: goalPoint,
        population_size: populationSize,
        max_evaluations: evaluationBudget,
        seed,
        ...(initialPopulation.length ? { initial_population: initialPopulation } : {}),
      });
      setJobId(job.job_id);
      setJobState({ job_id: job.job_id, status: 'running', elapsed_seconds: 0, algorithm, message: 'Optimizer is running' });
    } catch (error) {
      setJobState(null);
      setJobError((error as Error).message);
    }
  }

  function requestOptimization() {
    if (evaluationBudget >= 1000) setConfirmRun(true);
    else void beginOptimization();
  }

  async function stopOptimization() {
    if (!jobId || cancelling) return;
    try {
      const result = await cancelJob(jobId);
      setJobState(result);
      if (result.status === 'cancelled') setJobId(null);
    } catch (error) {
      setJobError((error as Error).message);
    }
  }

  function resetEnvironment() {
    setPlaying(false);
    setPlayback(0);
    setEventIndex(0);
    setStartPoint([0, 0, 0]);
    setGoalPoint([9, 9, 9]);
    setPreviousCandidate(null);
    setCurrentResult(null);
    setSelectedCandidateIndex(null);
    setCameraPreset('overview');
  }

  function moveEvent(index: number) {
    if (!scenario || running) return;
    const next = Math.max(0, Math.min(scenario.event_times.length - 1, index));
    if (selectedCandidate && next !== eventIndex) setPreviousCandidate(selectedCandidate);
    setPlaying(false);
    setPlayback(0);
    setEventIndex(next);
  }

  const eventDots = useMemo(() => scenario?.event_times ?? [], [scenario?.event_times]);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark"><Waypoints size={19} strokeWidth={1.6} /></div>
          <div className="brand-copy"><b>TRAJECTORY <span>LAB</span></b><small>AUTONOMOUS NAVIGATION RESEARCH</small></div>
        </div>
        <div className="topbar-context">
          <span className="context-label">SCENARIO</span>
          <b>{scenarioId === 'dynamic' ? 'CORRIDOR SHIFT / DYNAMIC' : 'OBSTACLE FIELD / STATIC'}</b>
          <span className="context-divider" />
          <span className="context-label">METHOD</span>
          <b>{selectedAlgorithm.short}</b>
        </div>
        <div className="topbar-actions">
          <div className={`system-state ${running ? 'system-running' : environmentHealth === 'ready' ? 'system-ready' : 'system-error'}`}>
            <span className="state-dot" />{running ? 'SOLVING' : environmentHealth === 'ready' ? 'SYSTEM READY' : environmentHealth === 'loading' ? 'CONNECTING' : 'API OFFLINE'}
          </div>
          <button className={`run-button ${running ? 'run-cancel' : ''}`} onClick={running ? stopOptimization : requestOptimization} disabled={environmentHealth !== 'ready' || !scenario || cancelling}>
            {running ? <><CircleStop size={15} /> {cancelling ? 'STOPPING' : 'STOP RUN'}</> : <><Play size={14} fill="currentColor" /> RUN PLAN</>}
          </button>
        </div>
      </header>

      <section className="workbench">
        <aside className="left-rail instrument-rail">
          <div className="rail-heading"><span className="eyebrow">01 / ENVIRONMENT</span><button className="icon-button small" title="Reset environment and view" onClick={resetEnvironment}><RotateCcw size={14} /></button></div>
          <section className="rail-section scenario-section">
            <div className="field-label">SCENARIO MODEL</div>
            <div className="segmented-control" role="group" aria-label="Scenario model">
              <button className={scenarioId === 'static' ? 'segment active' : 'segment'} disabled={running} onClick={() => { setScenarioId('static'); setEventIndex(0); }}>STATIC</button>
              <button className={scenarioId === 'dynamic' ? 'segment active' : 'segment'} disabled={running} onClick={() => { setScenarioId('dynamic'); setEventIndex(0); }}>DYNAMIC</button>
            </div>
            {scenarioId === 'dynamic' && scenario && (
              <div className="event-selector">
                <div className="field-label event-label">REPLANNING EVENT <span>T+{currentEvent.toFixed(1)} s</span></div>
                <div className="event-buttons">
                  {eventDots.map((time, index) => (
                    <button key={time} className={eventIndex === index ? 'event-button active' : 'event-button'} onClick={() => moveEvent(index)} disabled={running} title={`Environment event at ${time.toFixed(1)} seconds`}>
                      <span>{String(index + 1).padStart(2, '0')}</span><small>{time.toFixed(1)}s</small>
                    </button>
                  ))}
                </div>
              </div>
            )}
            {scenario && <div className="world-facts"><span><i /> {scenario.world.obstacles.length} obstacle{scenario.world.obstacles.length === 1 ? '' : 's'}</span><span>10 × 10 × 10 m</span></div>}
          </section>

          <section className="rail-section endpoint-section">
            <EndpointEditor start={startPoint} goal={goalPoint} disabled={running || environmentHealth === 'loading'} onApply={(start, goal) => { setStartPoint(start); setGoalPoint(goal); setPlaying(false); setPlayback(0); }} />
          </section>

          <section className="rail-section visibility-section">
            <div className="field-label">VIEW LAYERS</div>
            <Toggle checked={showObstacles} label="Obstacles" detail="Geometry + clearance shell" onChange={setShowObstacles} />
            <Toggle checked={showCandidates} label="Candidate paths" detail="Rank 0–2 population" onChange={setShowCandidates} />
            <Toggle checked={showControlPoints} label="Control points" detail="B-spline knots" onChange={setShowControlPoints} />
            <Toggle checked={showReference} label="Reference chord" detail="Start-to-goal line" onChange={setShowReference} />
          </section>

          <div className="rail-footnote"><ShieldAlert size={13} /><span>Amber paths are infeasible. Clearance envelope uses the backend safe-distance threshold.</span></div>
        </aside>

        <section className="simulation-column">
          <div className="viewport-shell">
            <div className="viewport-topline">
              <div className="viewport-coordinate"><span className="live-square" /> NAVIGATION VOLUME <b>3D · METRIC</b></div>
              <div className="viewport-actions">
                {CAMERA_PRESETS.map(({ id, label, icon: Icon }) => (
                  <button key={id} className={cameraPreset === id ? 'icon-button active' : 'icon-button'} onClick={() => setCameraPreset(id)} title={label} aria-label={label}><Icon size={15} /></button>
                ))}
                <span className="action-divider" />
                <button className={followVehicle ? 'icon-button active' : 'icon-button'} onClick={() => setFollowVehicle((value) => !value)} title="Follow vehicle" aria-label="Follow vehicle"><Orbit size={15} /></button>
                <button className={settingsOpen ? 'icon-button active' : 'icon-button'} onClick={() => setSettingsOpen((value) => !value)} title="Display settings" aria-label="Display settings"><Settings2 size={15} /></button>
                {settingsOpen && (
                  <div className="settings-popover">
                    <span className="eyebrow">RENDER QUALITY</span>
                    <div className="segmented-control compact"><button className={quality === 'balanced' ? 'segment active' : 'segment'} onClick={() => setQuality('balanced')}>BALANCED</button><button className={quality === 'high' ? 'segment active' : 'segment'} onClick={() => setQuality('high')}>HIGH</button></div>
                    <p>Reduced effects are respected automatically.</p>
                  </div>
                )}
              </div>
            </div>

            <div className="viewport-canvas" data-testid="simulation-canvas">
              {scenario ? <WorldScene
                scenario={scenario}
                selected={selectedCandidate}
                candidates={candidates}
                previous={previousCandidate}
                currentTime={currentTime}
                playing={playing}
                showObstacles={showObstacles}
                showCandidates={showCandidates}
                showControlPoints={showControlPoints}
                showReference={showReference}
                followVehicle={followVehicle}
                reduceMotion={reducedMotion}
                quality={quality}
                cameraPreset={cameraPreset}
                onSelectCandidate={setSelectedCandidateIndex}
              /> : <div className="scene-loading"><span className="loader-ring" />{environmentHealth === 'error' ? 'Python planner API unavailable' : 'Loading planning volume'}</div>}
              <div className="viewport-reticle" aria-hidden="true"><span /><span /><span /><span /></div>
              <div className="viewport-scale"><span>0</span><i /><span>10 m</span></div>
              <div className="viewport-top-right"><span className="event-chip">{scenarioId === 'dynamic' ? `EVENT ${String(eventIndex + 1).padStart(2, '0')}` : 'STATIC SNAPSHOT'}</span><span className="time-chip">T+{(currentEvent + currentTime).toFixed(2)} s</span></div>
              {scenarioError && <div className="viewport-error"><AlertTriangle size={16} /> {scenarioError}</div>}
              {visibleResult && <div className={`plan-state-chip ${selectedFeasible ? 'feasible' : 'warning'}`}><span />{selectedState}{selectedCandidate && ` · RANK ${String(selectedCandidate.rank).padStart(2, '0')}`}</div>}
            </div>

            <div className="playback-bar">
              <button className="playback-main" onClick={() => { if (!selectedCandidate) return; if (currentTime >= duration) setPlayback(0); setPlaying((value) => !value); }} disabled={!selectedCandidate} title={playing ? 'Pause trajectory playback' : 'Play trajectory playback'} aria-label={playing ? 'Pause trajectory playback' : 'Play trajectory playback'}>
                {playing ? <Pause size={16} fill="currentColor" /> : <Play size={16} fill="currentColor" />}
              </button>
              <button className="playback-reset" onClick={() => { setPlaying(false); setPlayback(0); }} disabled={!selectedCandidate} title="Reset trajectory playback" aria-label="Reset playback"><RotateCcw size={14} /></button>
              <span className="timecode">{timeLabel(currentTime)}</span>
              <div className="timeline-wrap">
                <input
                  aria-label="Trajectory playback time"
                  className="playback-range"
                  type="range"
                  min={0}
                  max={duration || 1}
                  step={0.01}
                  value={Math.min(currentTime, duration || 0)}
                  disabled={!selectedCandidate}
                  onChange={(event) => { setPlaying(false); setPlayback(Number(event.target.value)); }}
                />
                {scenarioId === 'dynamic' && scenario && <div className="timeline-events" aria-hidden="true">{scenario.event_times.map((time, index) => <span key={time} style={{ left: `${(time / duration) * 100}%` }} className={index === eventIndex ? 'active' : ''} />)}</div>}
              </div>
              <span className="timecode">{timeLabel(duration)}</span>
              <select className="speed-select" aria-label="Playback speed" value={playbackSpeed} onChange={(event) => setPlaybackSpeed(Number(event.target.value))}>
                <option value={0.5}>0.5×</option><option value={1}>1×</option><option value={2}>2×</option>
              </select>
              <button className="drawer-toggle" onClick={() => setDrawerOpen((value) => !value)} aria-expanded={drawerOpen} title={drawerOpen ? 'Collapse analysis panel' : 'Expand analysis panel'}><span>ANALYSIS</span>{drawerOpen ? <ArrowDown size={14} /> : <ChevronDown size={14} />}</button>
            </div>
          </div>

          <section className={`analysis-drawer ${drawerOpen ? 'open' : 'closed'}`}>
            {drawerOpen && <>
              <div className="analysis-tabs" role="tablist" aria-label="Analysis view">
                <button role="tab" aria-selected={activeTab === 'pareto'} className={activeTab === 'pareto' ? 'analysis-tab active' : 'analysis-tab'} onClick={() => setActiveTab('pareto')}><Waypoints size={14} /> PARETO SPACE</button>
                <button role="tab" aria-selected={activeTab === 'convergence'} className={activeTab === 'convergence' ? 'analysis-tab active' : 'analysis-tab'} onClick={() => setActiveTab('convergence')}><Activity size={14} /> CONVERGENCE</button>
                <button role="tab" aria-selected={activeTab === 'research'} className={activeTab === 'research' ? 'analysis-tab active' : 'analysis-tab'} onClick={() => setActiveTab('research')}><Gauge size={14} /> RESEARCH RECORD</button>
                <div className="analysis-note">OBJECTIVES ARE MINIMIZED · FRONT = TRADE-OFFS</div>
              </div>
              <div className="analysis-body">
                {activeTab === 'pareto' && <ParetoPlot candidates={candidates} selectedIndex={selectedCandidateIndex} onSelect={setSelectedCandidateIndex} />}
                {activeTab === 'convergence' && <ConvergencePlot result={visibleResult} />}
                {activeTab === 'research' && <ResearchPanel research={research} error={researchError} />}
              </div>
            </>}
          </section>
        </section>

        <aside className="right-rail instrument-rail">
          <div className="rail-heading"><span className="eyebrow">02 / SOLVER</span><span className="solver-stamp"><span /> PYTHON CORE</span></div>
          <section className="rail-section solver-section">
            <label className="field-label" htmlFor="algorithm-select">OPTIMIZATION METHOD</label>
            <div className="select-wrap"><select id="algorithm-select" value={algorithm} disabled={running} onChange={(event) => setAlgorithm(event.target.value as AlgorithmId)}>
              {ALGORITHMS.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}
            </select><ChevronDown size={14} /></div>
            <div className="method-caption">{algorithm === 'nsga2' ? 'Fast non-dominated sorting · SBX + polynomial mutation' : algorithm === 'mopso' ? 'Tchebycheff decomposition · external Pareto archive' : 'NSGA-II with Lamarckian elite refinement'}</div>
          </section>

          <section className="rail-section parameter-section">
            <div className="field-label">RUN PARAMETERS <span>EDITABLE</span></div>
            <RangeField label="Population size" value={populationSize} min={8} max={60} step={4} unit="members" onChange={setPopulationSize} disabled={running} />
            <RangeField label="Evaluation budget" value={evaluationBudget} min={80} max={1500} step={20} unit="evals" onChange={setEvaluationBudget} disabled={running} />
            <label className="number-field"><span>Random seed</span><input type="number" value={seed} min={0} max={2147483647} disabled={running} onChange={(event) => setSeed(Math.max(0, Number(event.target.value) || 0))} /></label>
            <div className="budget-note"><span>{evaluationBudget.toLocaleString()} OBJECTIVE EVALUATIONS</span><span>{populationSize} INITIAL MEMBERS</span></div>
            {canWarmStart && <label className={`warm-toggle ${warmStart ? 'checked' : ''}`}><input type="checkbox" checked={warmStart} disabled={running} onChange={(event) => setWarmStart(event.target.checked)} /><span className="checkmark">{warmStart && <Check size={12} />}</span><span><b>Warm-start previous population</b><small>All imported candidates are repaired and reevaluated.</small></span><Sparkles size={15} /></label>}
            {scenarioId === 'dynamic' && algorithm !== 'nsga2' && <div className="inline-note"><AlertTriangle size={13} /> Event-time motion is applied to this optimizer; population warm starts are NSGA-II only.</div>}
          </section>

          <button className="execute-button" onClick={running ? stopOptimization : requestOptimization} disabled={environmentHealth !== 'ready' || !scenario || cancelling}>
            {running ? <><CircleStop size={17} /> {cancelling ? 'STOPPING WORKER' : 'CANCEL OPTIMIZATION'}</> : <><Play size={16} fill="currentColor" /> EXECUTE {selectedAlgorithm.short}</>}
          </button>
          {running && <div className="run-progress-state"><span className="running-line" /><div><b>{cancelling ? 'Stopping optimizer worker' : 'Optimizer process active'}</b><small>{selectedAlgorithm.label} · {liveElapsed.toFixed(1)}s elapsed · {cancelling ? 'Waiting for process termination' : 'cancel stops the worker process'}</small></div></div>}
          {jobError && <div className="error-banner" role="alert"><AlertTriangle size={14} /><span>{jobError}</span><button onClick={() => setJobError(null)} aria-label="Dismiss error"><X size={13} /></button></div>}

          <div className="metrics-heading"><span className="eyebrow">03 / CURRENT SOLUTION</span><span className={`metric-status ${selectedStatusClass}`}>{selectedState}</span></div>
          <section className="metric-section">
            <div className="candidate-select-row">
              <span>{visibleResult ? `CANDIDATE ${String((selectedCandidate?.index ?? 0) + 1).padStart(2, '0')} / ${String(candidates.length).padStart(2, '0')}` : 'NO OPTIMIZATION RESULT'}</span>
              <div><button className="icon-button tiny" title="Previous candidate" aria-label="Previous candidate" disabled={!candidates.length} onClick={() => cycleCandidate(-1)}><ArrowLeft size={13} /></button><button className="icon-button tiny" title="Next candidate" aria-label="Next candidate" disabled={!candidates.length} onClick={() => cycleCandidate(1)}><ArrowRight size={13} /></button></div>
            </div>
            <MetricRow label="Path length" value={objectiveLabel(selectedCandidate?.objectives.path_length, 2)} unit="m" tone="cyan" />
            <MetricRow label="Clearance risk" value={objectiveLabel(selectedCandidate?.objectives.clearance_risk, 4)} unit="obj" tone="amber" />
            <MetricRow label="Minimum snap" value={objectiveLabel(selectedCandidate?.objectives.minimum_snap, 4)} unit="obj" tone="violet" />
            <MetricRow label="Minimum clearance" value={objectiveLabel(selectedCandidate?.minimum_clearance, 3)} unit="m raw" tone={selectedCandidate && selectedClearanceMargin != null && selectedClearanceMargin < 0 ? 'amber' : undefined} />
            <MetricRow label="Safety margin" value={objectiveLabel(selectedClearanceMargin, 3)} unit={`m vs ${scenario?.world.d_safe.toFixed(2) ?? '—'}`} tone={selectedClearanceMargin != null && selectedClearanceMargin < 0 ? 'amber' : undefined} />
            <div className="metric-divider" />
            <MetricRow label="Feasible members" value={visibleResult ? `${feasibleCount} / ${candidates.length}` : '—'} unit={visibleResult ? `${(visibleResult.feasibility_rate * 100).toFixed(0)}%` : ''} />
            <MetricRow label="Constraint violation" value={objectiveLabel(selectedCandidate?.total_violation, 4)} unit={selectedFeasible ? 'none' : 'sum'} tone={selectedCandidate && !selectedFeasible ? 'amber' : undefined} />
            <MetricRow label="Velocity · peak" value={objectiveLabel(selectedCandidate?.max_speed, 3)} unit="m/s" />
            <MetricRow label="Acceleration · peak" value={objectiveLabel(selectedCandidate?.max_acceleration, 3)} unit="m/s²" />
            <div className="metric-divider" />
            <MetricRow label="Objective evaluations" value={visibleResult?.objective_evaluations.toLocaleString() ?? '—'} unit={visibleResult ? `/${evaluationBudget}` : ''} />
            <MetricRow label="Optimizer time" value={visibleResult ? visibleResult.optimization_seconds.toFixed(3) : running ? liveElapsed.toFixed(1) : '—'} unit="s" />
            {visibleResult && visibleResult.imported_trajectory_count > 0 && <MetricRow label="Imported · reevaluated" value={`${visibleResult.accepted_imported_count} · ${visibleResult.reevaluated_imported_count}`} unit={`${visibleResult.accepted_imported_feasible_count} feasible`} tone="violet" />}
            {visibleResult && <div className={`constraint-summary ${selectedFeasible ? 'good' : 'bad'}`}><span />{selectedFeasible ? 'All hard constraints satisfied' : 'Constraint review required'}<small>{selectedCandidate?.violation_details.collision != null ? `COLLISION ${selectedCandidate.violation_details.collision.toFixed(3)} · ` : ''}V {selectedCandidate?.violation_details.velocity?.toFixed(3) ?? '0.000'} · A {selectedCandidate?.violation_details.acceleration?.toFixed(3) ?? '0.000'} · B {selectedCandidate?.violation_details.bounds?.toFixed(3) ?? '0.000'}</small></div>}
          </section>
          <div className="right-footnote"><Timer size={13} /><span>{running ? 'Live time is wall-clock elapsed; no synthetic progress is shown.' : visibleResult ? `${visibleResult.generation_count} recorded generations · ${visibleResult.objective_evaluations} evaluations` : 'Choose a method and execute a planner run.'}</span></div>
        </aside>
      </section>

      <footer className="statusbar">
        <span><i className={environmentHealth === 'ready' ? 'online' : ''} />{environmentHealth === 'ready' ? 'LOCAL PYTHON PLANNER CONNECTED' : environmentHealth === 'error' ? 'PLANNER API CONNECTION FAILED' : 'CONNECTING TO PLANNER'}</span>
        <span>DEGREE-5 CLAMPED B-SPLINE</span>
        <span>{scenario ? `${scenario.world.v_max.toFixed(1)} m/s MAX · ${scenario.world.a_max.toFixed(1)} m/s² MAX` : '—'}</span>
        <button aria-pressed={reducedMotion} onClick={() => setReducedMotion((value) => !value)} title="Toggle reduced motion" aria-label="Toggle reduced motion"><SlidersHorizontal size={13} /> MOTION</button>
      </footer>

      {confirmRun && <div className="modal-backdrop" role="presentation"><section className="confirm-modal" role="dialog" aria-modal="true" aria-labelledby="confirm-title"><div className="modal-icon"><AlertTriangle size={19} /></div><span className="eyebrow">COMPUTE CONFIRMATION</span><h2 id="confirm-title">Start a larger solve?</h2><p>This run requests {evaluationBudget.toLocaleString()} objective evaluations. The Python worker can be stopped at any time from the run controls.</p><div className="modal-actions"><button className="secondary-button" onClick={() => setConfirmRun(false)}>KEEP EDITING</button><button className="run-button" onClick={() => void beginOptimization()}><Play size={14} fill="currentColor" /> START SOLVE</button></div></section></div>}
    </main>
  );

  function cycleCandidate(direction: number) {
    if (!candidates.length) return;
    const current = candidates.findIndex((candidate) => candidate.index === selectedCandidateIndex);
    const next = (current + direction + candidates.length) % candidates.length;
    setSelectedCandidateIndex(candidates[next].index);
  }
}

function EndpointEditor({
  start,
  goal,
  disabled,
  onApply,
}: {
  start: Vec3;
  goal: Vec3;
  disabled: boolean;
  onApply: (start: Vec3, goal: Vec3) => void;
}) {
  const [draftStart, setDraftStart] = useState<Vec3>(start);
  const [draftGoal, setDraftGoal] = useState<Vec3>(goal);
  const [draftError, setDraftError] = useState<string | null>(null);
  const startKey = start.join(',');
  const goalKey = goal.join(',');

  useEffect(() => {
    setDraftStart(start);
    setDraftGoal(goal);
    setDraftError(null);
  }, [startKey, goalKey]);

  const changed = draftStart.some((value, axis) => value !== start[axis])
    || draftGoal.some((value, axis) => value !== goal[axis]);

  function apply() {
    if (![...draftStart, ...draftGoal].every((value) => Number.isFinite(value) && value >= 0 && value <= 10)) {
      setDraftError('Coordinates must remain inside the 0–10 m world bounds.');
      return;
    }
    if (draftStart.every((value, axis) => value === draftGoal[axis])) {
      setDraftError('Origin and goal must be different points.');
      return;
    }
    setDraftError(null);
    onApply(draftStart, draftGoal);
  }

  function coordinateRow(label: string, marker: string, values: Vec3, update: (value: Vec3) => void, origin: boolean) {
    return (
      <div className="coordinate-row">
        <span className={`waypoint-symbol ${origin ? 'origin' : 'goal'}`}>{marker}</span>
        <div className="coordinate-row-fields">
          <small>{label}</small>
          <div className="coordinate-inputs">
            {(['X', 'Y', 'Z'] as const).map((axis, index) => (
              <label key={axis}><span>{axis}</span><input aria-label={`${label} ${axis}`} type="number" min={0} max={10} step={0.5} value={values[index]} disabled={disabled} onChange={(event) => update(values.map((value, pointIndex) => pointIndex === index ? Number(event.target.value) : value) as Vec3)} /></label>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <>
      <div className="field-label">WAYPOINT CONFIGURATION <span>XYZ · m</span></div>
      {coordinateRow('ORIGIN', 'O', draftStart, setDraftStart, true)}
      <div className="waypoint-link" />
      {coordinateRow('GOAL', 'G', draftGoal, setDraftGoal, false)}
      <div className="endpoint-apply-row">
        {draftError && <span role="alert">{draftError}</span>}
        <button className={changed ? 'apply-waypoints dirty' : 'apply-waypoints'} disabled={disabled || !changed} onClick={apply}><Check size={12} /> APPLY</button>
      </div>
    </>
  );
}

function Toggle({ checked, label, detail, onChange }: { checked: boolean; label: string; detail: string; onChange: (value: boolean) => void }) {
  return <label className="view-toggle"><input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} /><span className="toggle-switch" /><span className="toggle-copy"><b>{label}</b><small>{detail}</small></span></label>;
}

function RangeField({ label, value, min, max, step, unit, onChange, disabled }: { label: string; value: number; min: number; max: number; step: number; unit: string; onChange: (value: number) => void; disabled: boolean }) {
  return <label className="range-field"><span><b>{label}</b><small>{value} {unit}</small></span><input type="range" min={min} max={max} step={step} value={value} disabled={disabled} onChange={(event) => onChange(Number(event.target.value))} /></label>;
}

function MetricRow({ label, value, unit, tone }: { label: string; value: string; unit?: string; tone?: string }) {
  return <div className={`metric-row ${tone ?? ''}`}><span>{label}</span><b>{value}</b><small>{unit}</small></div>;
}
