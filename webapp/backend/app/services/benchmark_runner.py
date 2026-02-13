"""Spawn pipeline/orchestrator.py as a subprocess and stream output."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..models.schemas import BenchmarkRequest, BenchmarkStatus

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Manages benchmark subprocess lifecycle."""

    def __init__(self, lab_root: Path, *, results_loader: Any = None) -> None:
        self.lab_root = lab_root
        self.orchestrator = lab_root / "pipeline" / "orchestrator.py"
        self._results_loader = results_loader  # set post-init from main.py
        self._jobs: dict[str, BenchmarkStatus] = {}
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._log_buffers: dict[str, list[str]] = {}
        self._subscribers: dict[str, list[asyncio.Queue[str]]] = {}
        self._finished: set[str] = set()  # job_ids whose subprocess has exited
        self._active_job: str | None = None

        # Persistent storage for logs and job metadata
        self._logs_dir = lab_root / "logs"
        self._jobs_dir = self._logs_dir / "jobs"
        self._jobs_dir.mkdir(parents=True, exist_ok=True)

        # Restore job history from disk
        self._restore_jobs()

    # ------------------------------------------------------------------
    # Helpers — Python path
    # ------------------------------------------------------------------

    def _python_executable(self) -> str:
        """Return the lab venv python3, falling back to system python3."""
        venv_python = self.lab_root / ".venv" / "bin" / "python3"
        if venv_python.is_file():
            return str(venv_python)
        logger.warning(
            "Lab venv not found at %s — falling back to system python3. "
            "The orchestrator may fail if numpy/pyerfa are not installed.",
            venv_python,
        )
        return "python3"

    # ------------------------------------------------------------------
    # Job persistence
    # ------------------------------------------------------------------

    def _persist_job(self, job_id: str) -> None:
        """Write job status to disk so it survives server restarts."""
        status = self._jobs.get(job_id)
        if status is None:
            return
        path = self._jobs_dir / f"{job_id}.json"
        path.write_text(status.model_dump_json(indent=2))

    def _restore_jobs(self) -> None:
        """Reload job metadata from logs/jobs/*.json on startup."""
        for p in sorted(self._jobs_dir.glob("*.json")):
            try:
                data = json.loads(p.read_text())
                status = BenchmarkStatus(**data)
                self._jobs[status.job_id] = status
                self._finished.add(status.job_id)
                # Restore log buffer from the log file (if it exists)
                log_file = self._log_file_path(status.job_id)
                if log_file.is_file():
                    self._log_buffers[status.job_id] = (
                        log_file.read_text().splitlines()
                    )
            except Exception as exc:
                logger.warning("Failed to restore job from %s: %s", p, exc)

    def _log_file_path(self, job_id: str) -> Path:
        return self._logs_dir / f"{job_id}.log"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_busy(self) -> bool:
        return self._active_job is not None

    def get_status(self, job_id: str) -> BenchmarkStatus | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[BenchmarkStatus]:
        return list(self._jobs.values())

    def get_log_lines(self, job_id: str) -> list[str] | None:
        """Return buffered log lines for a job, or None if unknown."""
        if job_id not in self._jobs:
            return None
        return self._log_buffers.get(job_id, [])

    def is_finished(self, job_id: str) -> bool:
        return job_id in self._finished

    async def start(self, request: BenchmarkRequest) -> BenchmarkStatus:
        """Start a new benchmark run. Returns the job status."""
        if self._active_job is not None:
            raise RuntimeError("A benchmark is already running")

        job_id = uuid.uuid4().hex[:12]
        status = BenchmarkStatus(
            job_id=job_id,
            status="running",
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        self._jobs[job_id] = status
        self._log_buffers[job_id] = []
        self._subscribers[job_id] = []
        self._active_job = job_id

        # Persist initial status
        self._persist_job(job_id)

        # Build command — use lab venv python, not bare python3
        python = self._python_executable()
        cmd = [python, str(self.orchestrator)]

        # Join experiments with commas for the --experiments flag
        exps = ",".join(request.experiments)
        cmd.extend(["--experiments", exps])
        cmd.extend(["--n", str(request.n)])
        cmd.extend(["--seed", str(request.seed)])
        if request.no_perf:
            cmd.append("--no-perf")

        logger.info("Starting benchmark: %s", " ".join(cmd))

        # Spawn
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(self.lab_root),
        )
        self._processes[job_id] = proc

        # Read output in background
        asyncio.create_task(self._read_output(job_id, proc))
        return status

    def subscribe(self, job_id: str) -> asyncio.Queue[str]:
        """Get a queue that will receive log lines for a job."""
        q: asyncio.Queue[str] = asyncio.Queue()
        if job_id not in self._subscribers:
            self._subscribers[job_id] = []
        self._subscribers[job_id].append(q)
        # Replay buffered lines
        for line in self._log_buffers.get(job_id, []):
            q.put_nowait(line)
        # If the job already finished, send __EOF__ immediately
        if job_id in self._finished:
            q.put_nowait("__EOF__")
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue[str]) -> None:
        subs = self._subscribers.get(job_id, [])
        if q in subs:
            subs.remove(q)

    async def cancel(self, job_id: str) -> bool:
        """Cancel a running benchmark job.
        
        Returns:
            True if the job was cancelled, False if it wasn't running.
        """
        # Check if this job is actually running
        if job_id not in self._processes or job_id in self._finished:
            return False

        proc = self._processes.get(job_id)
        if proc is None or proc.returncode is not None:
            return False

        logger.info("Cancelling job %s", job_id)

        # Terminate the process gracefully
        try:
            proc.terminate()
            # Wait briefly for graceful shutdown
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                # Force kill if it doesn't terminate
                proc.kill()
                await proc.wait()
        except ProcessLookupError:
            # Process already died
            pass

        # Update status
        status = self._jobs.get(job_id)
        if status is not None:
            status.status = "cancelled"
            status.finished_at = datetime.now(timezone.utc).isoformat()
            self._persist_job(job_id)

        # Clean up active job tracking
        if self._active_job == job_id:
            self._active_job = None
        self._finished.add(job_id)

        # Notify subscribers
        for q in self._subscribers.get(job_id, []):
            try:
                q.put_nowait("__CANCELLED__")
                q.put_nowait("__EOF__")
            except asyncio.QueueFull:
                pass

        logger.info("Job %s cancelled successfully", job_id)
        return True

    async def _read_output(
        self, job_id: str, proc: asyncio.subprocess.Process
    ) -> None:
        """Read subprocess stdout line by line and dispatch to subscribers."""
        log_file = self._log_file_path(job_id)
        try:
            with open(log_file, "w") as fh:
                assert proc.stdout is not None
                async for raw_line in proc.stdout:
                    line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
                    # Buffer in memory
                    self._log_buffers.setdefault(job_id, []).append(line)
                    # Persist to disk
                    fh.write(line + "\n")
                    fh.flush()
                    # Dispatch to live subscribers
                    for q in self._subscribers.get(job_id, []):
                        try:
                            q.put_nowait(line)
                        except asyncio.QueueFull:
                            pass

            await proc.wait()
        finally:
            # Update status (unless already set to cancelled)
            status = self._jobs.get(job_id)
            if status is not None and status.status != "cancelled":
                status.status = "completed" if proc.returncode == 0 else "failed"
                status.finished_at = datetime.now(timezone.utc).isoformat()
            self._active_job = None
            self._finished.add(job_id)

            # Persist final status to disk
            self._persist_job(job_id)

            # Reload the results cache unconditionally so new runs are visible
            # even if no WebSocket client is connected
            if self._results_loader is not None:
                try:
                    self._results_loader.reload()
                    logger.info("Results cache reloaded after job %s", job_id)
                except Exception as exc:
                    logger.warning("Failed to reload results: %s", exc)

            # Signal end to any current subscribers
            for q in self._subscribers.get(job_id, []):
                try:
                    q.put_nowait("__EOF__")
                except asyncio.QueueFull:
                    pass
