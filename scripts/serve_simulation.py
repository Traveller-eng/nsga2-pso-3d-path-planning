from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
WEB_DIST = ROOT / "web" / "dist"
WORKER = ROOT / "scripts" / "simulation_worker.py"
RESULTS = ROOT / "experiments" / "results"
JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
MAX_REQUEST_BYTES = 1_000_000


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def research_payload() -> dict[str, Any]:
    benchmark = read_json(RESULTS / "benchmark_runs.json")
    dynamic_summary = read_json(RESULTS / "dynamic_warm_start_summary.json")
    dynamic_analysis = read_json(RESULTS / "dynamic_warm_start_analysis.json")

    benchmark_runs = benchmark.get("runs", [])
    algorithms: dict[str, list[dict[str, Any]]] = {}
    for run in benchmark_runs:
        algorithms.setdefault(run["algorithm"], []).append(run)
    static_summary = []
    for algorithm, runs in algorithms.items():
        static_summary.append({
            "algorithm": algorithm,
            "seed_count": len(runs),
            "evaluation_budget": int(benchmark["metadata"]["configuration"]["max_evaluations"]),
            "mean_runtime_seconds": sum(float(run["runtime"]) for run in runs) / len(runs),
            "mean_hypervolume": sum(float(run["hypervolume"]) for run in runs) / len(runs),
            "mean_igd": sum(float(run["igd"]) for run in runs) / len(runs),
            "mean_feasible_count": sum(float(run["feasible_count"]) for run in runs) / len(runs),
        })

    by_event: dict[str, dict[str, Any]] = {}
    for row in dynamic_summary["condition_summary_by_event"]:
        event = by_event.setdefault(str(row["event_index"]), {"event_time": row["event_time"], "conditions": {}})
        event["conditions"].setdefault(row["condition"], {})[row["metric"]] = {
            key: row[key] for key in ("n", "mean", "median", "std_sample", "mean_ci95")
        }
    dynamic_differences = [
        row for row in dynamic_analysis["paired_tests"]["paired_differences"]
        if row["metric"] in dynamic_analysis["paired_tests"]["primary_metrics"]
    ]
    return {
        "static_benchmark": {
            "configuration": benchmark["metadata"]["configuration_file"],
            "objective_space": benchmark["metadata"]["objective_space"],
            "scenario": "Static, obstacle-free reference world from the frozen benchmark configuration.",
            "common_reference_front_size": benchmark["metadata"]["igd_reference_front_size"],
            "algorithms": static_summary,
        },
        "dynamic_replanning": {
            "configuration": "experiments/configs/dynamic_warm_start.json",
            "paired_seed_count": len(dynamic_analysis["seeds"]),
            "budget_per_event": 100,
            "event_count": len(by_event),
            "events": by_event,
            "paired_differences": dynamic_differences,
            "interpretation": "The stored results show no adjusted evidence of reduced replanning latency. Final-event feasibility and hypervolume are descriptively higher for warm-start, but the corrected t-test and Wilcoxon results disagree and the cold condition has no feasible final-event run. Results are scenario-specific.",
        },
    }


def job_snapshot(job: dict[str, Any]) -> dict[str, Any]:
    process = job["process"]
    return_code = process.poll()
    if return_code is None:
        cancelling = bool(job["cancel_requested"])
        return {
            "job_id": job["job_id"],
            "status": "cancelling" if cancelling else "running",
            "algorithm": job["request"]["algorithm"],
            "elapsed_seconds": time.perf_counter() - job["started"],
            "message": "Stopping optimizer worker." if cancelling else "Optimizer is running; generation details appear when the backend returns its recorded history.",
        }

    if job["cancel_requested"]:
        job["status"] = "cancelled"
        job["error"] = "Optimization was cancelled by the operator."
    elif return_code == 0 and job["output_path"].exists():
        try:
            job["result"] = read_json(job["output_path"])
            job["status"] = "complete"
        except (OSError, json.JSONDecodeError) as exc:
            job["status"] = "failed"
            job["error"] = f"Unable to read worker result: {exc}"
    else:
        job["status"] = "failed"
        try:
            detail = job["log_path"].read_text(encoding="utf-8", errors="replace")[-3000:]
        except OSError:
            detail = "Worker exited without a result file."
        job["error"] = detail.strip() or f"Worker exited with code {return_code}."

    if job["status"] == "complete":
        return {"job_id": job["job_id"], "status": "complete", "result": job["result"]}
    return {"job_id": job["job_id"], "status": job["status"], "error": job["error"]}


class SimulationHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIST), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self.send_json(200, {"status": "ok", "planner": "python", "frontend_built": (WEB_DIST / "index.html").exists()})
        elif parsed.path == "/api/scenario":
            from planner.simulation import scenario_payload

            query = parse_qs(parsed.query)
            scenario_id = query.get("scenario", ["static"])[0]
            try:
                event_index = int(query.get("event", ["0"])[0])
                def query_point(name: str) -> tuple[float, ...] | None:
                    value = query.get(name, [None])[0]
                    if value is None:
                        return None
                    coordinates = tuple(float(part) for part in value.split(","))
                    if len(coordinates) != 3:
                        raise ValueError(f"{name} must contain three comma-separated coordinates.")
                    return coordinates

                self.send_json(200, scenario_payload(scenario_id, event_index, query_point("start"), query_point("goal")))
            except (ValueError, KeyError) as exc:
                self.send_json(400, {"error": str(exc)})
        elif parsed.path == "/api/research":
            try:
                self.send_json(200, research_payload())
            except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
                self.send_json(500, {"error": f"Research artifact unavailable: {exc}"})
        elif parsed.path.startswith("/api/jobs/"):
            job_id = parsed.path.rsplit("/", 1)[-1]
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if job is None:
                    self.send_json(404, {"error": "Unknown optimization job."})
                else:
                    self.send_json(200, job_snapshot(job))
        elif parsed.path.startswith("/api/"):
            self.send_json(404, {"error": "Unknown API endpoint."})
        else:
            if not (WEB_DIST / "index.html").exists():
                self.send_error(503, "Frontend build is missing. Start Vite or run npm run build in web/.")
                return
            requested = parsed.path.lstrip("/")
            target = (WEB_DIST / requested).resolve()
            try:
                target.relative_to(WEB_DIST.resolve())
            except ValueError:
                self.send_error(403)
                return
            if not target.exists() or target.is_dir():
                self.path = "/index.html"
            super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/api/jobs":
            self.send_json(404, {"error": "Unknown API endpoint."})
            return
        try:
            request = self.read_request_json()
            self.send_json(202, self.start_job(request))
        except (ValueError, KeyError, OSError) as exc:
            self.send_json(400, {"error": str(exc)})
        except RuntimeError as exc:
            self.send_json(409, {"error": str(exc)})

    def do_DELETE(self) -> None:
        if not self.path.startswith("/api/jobs/"):
            self.send_json(404, {"error": "Unknown API endpoint."})
            return
        job_id = self.path.rsplit("/", 1)[-1]
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job is None:
                self.send_json(404, {"error": "Unknown optimization job."})
                return
            if job["process"].poll() is None:
                job["cancel_requested"] = True
                job["process"].terminate()
            self.send_json(200, job_snapshot(job))

    def read_request_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_REQUEST_BYTES:
            raise ValueError("Request body must be between 1 byte and 1 MB.")
        payload = json.loads(self.rfile.read(length))
        if not isinstance(payload, dict):
            raise ValueError("Request must be a JSON object.")
        return payload

    def start_job(self, request: dict[str, Any]) -> dict[str, Any]:
        required = {"algorithm", "scenario_id", "event_index", "population_size", "max_evaluations", "seed"}
        missing = sorted(required - request.keys())
        if missing:
            raise ValueError(f"Missing request fields: {', '.join(missing)}")
        if request["scenario_id"] not in {"static", "dynamic"}:
            raise ValueError("scenario_id must be static or dynamic.")
        if request["algorithm"] not in {"nsga2", "mopso", "hybrid"}:
            raise ValueError("algorithm must be nsga2, mopso, or hybrid.")
        initial_population = request.get("initial_population", [])
        if initial_population and request["algorithm"] != "nsga2":
            raise ValueError("Warm starts are supported only for NSGA-II.")
        if len(initial_population) > int(request["population_size"]):
            raise ValueError("Initial population cannot exceed the configured population size.")

        with JOBS_LOCK:
            for existing in JOBS.values():
                if existing["process"].poll() is None:
                    raise RuntimeError("An optimization is already running. Cancel it or wait for its result.")

            job_id = uuid.uuid4().hex[:12]
            workdir = Path(tempfile.mkdtemp(prefix=f"planner-{job_id}-"))
            request_path = workdir / "request.json"
            output_path = workdir / "result.json"
            log_path = workdir / "worker.log"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            environment = os.environ.copy()
            source_path = str(ROOT / "src")
            environment["PYTHONPATH"] = source_path + os.pathsep + environment.get("PYTHONPATH", "")
            with log_path.open("w", encoding="utf-8") as log_file:
                process = subprocess.Popen(
                    [sys.executable, str(WORKER), "--request", str(request_path), "--output", str(output_path)],
                    cwd=ROOT,
                    env=environment,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
            JOBS[job_id] = {
                "job_id": job_id,
                "request": request,
                "process": process,
                "started": time.perf_counter(),
                "workdir": workdir,
                "output_path": output_path,
                "log_path": log_path,
                "cancel_requested": False,
                "status": "running",
                "error": None,
                "result": None,
            }
        return {"job_id": job_id, "status": "running", "algorithm": request["algorithm"]}

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")


def main() -> None:
    from planner.simulation import scenario_payload

    address = ("127.0.0.1", int(os.environ.get("PLANNER_PORT", "8000")))
    server = ThreadingHTTPServer(address, SimulationHandler)
    print(f"Planner API and built frontend: http://{address[0]}:{address[1]}")
    print("Development UI: run `cd web && npm run dev` (Vite proxies /api to port 8000).")
    print(f"Scenario endpoint smoke: {json.dumps(scenario_payload('dynamic', 0)['world']['start'])}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        with JOBS_LOCK:
            for job in JOBS.values():
                if job["process"].poll() is None:
                    job["process"].terminate()
        server.server_close()


if __name__ == "__main__":
    main()
