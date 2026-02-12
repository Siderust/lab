"""Benchmark execution API — start runs and stream logs."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

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


@router.websocket("/benchmark/ws/{job_id}")
async def benchmark_ws(websocket: WebSocket, job_id: str) -> None:
    from ..main import loader, runner

    status = runner.get_status(job_id)
    if status is None:
        await websocket.close(code=4004, reason="Job not found")
        return

    await websocket.accept()
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
                # Reload results so the new run is visible
                loader.reload()
                break
            await websocket.send_text(json.dumps({"type": "log", "line": line}))
    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass
    finally:
        runner.unsubscribe(job_id, q)
