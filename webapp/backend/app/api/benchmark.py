"""Benchmark execution API — start runs and stream logs."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse

from ..models.schemas import BenchmarkRequest, BenchmarkStatus

router = APIRouter(tags=["benchmark"])


@router.post("/benchmark/start", response_model=BenchmarkStatus)
async def start_benchmark(request: BenchmarkRequest) -> BenchmarkStatus:
    from ..main import runner

    if runner.is_busy:
        raise HTTPException(status_code=409, detail="A benchmark is already running")
    status = await runner.start(request)
    return status


@router.get("/benchmark/jobs", response_model=list[BenchmarkStatus])
async def list_jobs() -> list[BenchmarkStatus]:
    from ..main import runner

    return runner.list_jobs()


@router.get("/benchmark/jobs/{job_id}", response_model=BenchmarkStatus)
async def get_job(job_id: str) -> BenchmarkStatus:
    from ..main import runner

    status = runner.get_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return status


@router.post("/benchmark/jobs/{job_id}/cancel", response_model=BenchmarkStatus)
async def cancel_job(job_id: str) -> BenchmarkStatus:
    """Cancel a running benchmark job."""
    from ..main import runner

    status = runner.get_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    if status.status not in ["running", "starting"]:
        raise HTTPException(
            status_code=409,
            detail=f"Job '{job_id}' is not running (status: {status.status})"
        )

    cancelled = await runner.cancel(job_id)
    if not cancelled:
        raise HTTPException(status_code=500, detail="Failed to cancel job")

    # Return updated status
    updated_status = runner.get_status(job_id)
    if updated_status is None:
        raise HTTPException(status_code=500, detail="Job status lost after cancellation")
    return updated_status


@router.get("/benchmark/jobs/{job_id}/logs")
async def get_job_logs(job_id: str) -> PlainTextResponse:
    """Return the full log output for a job (from disk or memory buffer)."""
    from ..main import runner

    lines = runner.get_log_lines(job_id)
    if lines is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return PlainTextResponse("\n".join(lines))


@router.websocket("/benchmark/ws/{job_id}")
async def benchmark_ws(websocket: WebSocket, job_id: str) -> None:
    from ..main import loader, runner

    status = runner.get_status(job_id)
    if status is None:
        await websocket.close(code=4004, reason="Job not found")
        return

    await websocket.accept()

    # If the job already finished before WS connected, send buffered logs + done
    if runner.is_finished(job_id):
        for line in runner.get_log_lines(job_id) or []:
            await websocket.send_text(json.dumps({"type": "log", "line": line}))
        final = runner.get_status(job_id)
        await websocket.send_text(
            json.dumps({"type": "done", "status": final.status if final else "unknown"})
        )
        return

    q = runner.subscribe(job_id)

    try:
        while True:
            line = await asyncio.wait_for(q.get(), timeout=300)
            if line == "__EOF__":
                # Send final status
                final = runner.get_status(job_id)
                await websocket.send_text(
                    json.dumps({"type": "done", "status": final.status if final else "unknown"})
                )
                # Belt-and-suspenders reload (main reload is in _read_output)
                loader.reload()
                break
            await websocket.send_text(json.dumps({"type": "log", "line": line}))
    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass
    finally:
        runner.unsubscribe(job_id, q)
