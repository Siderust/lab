"""Experiment detail API."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..models.schemas import ExperimentResult

router = APIRouter(tags=["experiments"])


@router.get(
    "/runs/{run_id}/experiments/{experiment}",
    response_model=list[ExperimentResult],
)
async def get_experiment(run_id: str, experiment: str) -> list[ExperimentResult]:
    from ..main import loader

    results = loader.get_experiment(run_id, experiment)
    if results is None:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment '{experiment}' not found in run '{run_id}'",
        )
    return results
