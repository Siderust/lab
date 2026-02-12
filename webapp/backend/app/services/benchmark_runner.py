"""Spawn pipeline/orchestrator.py as a subprocess and stream output."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..models.schemas import BenchmarkRequest, BenchmarkStatus


class BenchmarkRunner:
    """Manages benchmark subprocess lifecycle."""

    def __init__(self, lab_root: Path) -> None:
        self.lab_root = lab_root
        self.orchestrator = lab_root / "pipeline" / "orchestrator.py"
        self._jobs: dict[str, BenchmarkStatus] = {}
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._log_buffers: dict[str, list[str]] = {}
        self._subscribers: dict[str, list[asyncio.Queue[str]]] = {}
        self._active_job: str | None = None

    @property
    def is_busy(self) -> bool:
        return self._active_job is not None

    def get_status(self, job_id: str) -> BenchmarkStatus | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[BenchmarkStatus]:
        return list(self._jobs.values())

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

        # Build command
        cmd = ["python3", str(self.orchestrator)]
        for exp in request.experiments:
            cmd.extend(["--experiment", exp])
        cmd.extend(["--n", str(request.n)])
        cmd.extend(["--seed", str(request.seed)])
        if request.no_perf:
            cmd.append("--no-perf")

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
        # Send buffered lines
        for line in self._log_buffers.get(job_id, []):
            q.put_nowait(line)
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue[str]) -> None:
        subs = self._subscribers.get(job_id, [])
        if q in subs:
            subs.remove(q)

    async def _read_output(
        self, job_id: str, proc: asyncio.subprocess.Process
    ) -> None:
        """Read subprocess stdout line by line and dispatch to subscribers."""
        try:
            assert proc.stdout is not None
            async for raw_line in proc.stdout:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
                self._log_buffers.setdefault(job_id, []).append(line)
                for q in self._subscribers.get(job_id, []):
                    try:
                        q.put_nowait(line)
                    except asyncio.QueueFull:
                        pass

            await proc.wait()
        finally:
            status = self._jobs.get(job_id)
            if status is not None:
                status.status = "completed" if proc.returncode == 0 else "failed"
                status.finished_at = datetime.now(timezone.utc).isoformat()
            self._active_job = None

            # Signal end to subscribers
            for q in self._subscribers.get(job_id, []):
                try:
                    q.put_nowait("__EOF__")
                except asyncio.QueueFull:
                    pass
