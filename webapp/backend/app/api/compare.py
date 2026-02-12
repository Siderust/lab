"""Compare two runs."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(tags=["compare"])


@router.get("/compare")
async def compare_runs(
    run_a: str = Query(..., description="First run ID (date)"),
    run_b: str = Query(..., description="Second run ID (date)"),
) -> dict[str, Any]:
    from ..main import loader
    from ..services.results_loader import compute_comparison

    if loader.get_run(run_a) is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_a}' not found")
    if loader.get_run(run_b) is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_b}' not found")

    deltas = compute_comparison(loader, run_a, run_b)
    return {"run_a": run_a, "run_b": run_b, "deltas": deltas}
