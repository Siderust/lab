"""Runs API — list and retrieve benchmark runs."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..models.schemas import RunDetail, RunSummary

router = APIRouter(tags=["runs"])


@router.get("/runs", response_model=list[RunSummary])
async def list_runs() -> list[RunSummary]:
    from ..main import loader

    return loader.list_runs()


@router.get("/runs/{run_id}", response_model=RunDetail)
async def get_run(run_id: str) -> RunDetail:
    from ..main import loader

    detail = loader.get_run(run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return detail


@router.post("/runs/reload")
async def reload_runs() -> dict[str, str]:
    """Force re-scan of results/ and reports/ directories."""
    from ..main import loader

    loader.reload()
    return {"status": "ok"}
