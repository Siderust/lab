"""Siderust Lab — FastAPI backend."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api import benchmark, compare, experiments, runs
from .services.benchmark_runner import BenchmarkRunner
from .services.results_loader import ResultsLoader

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _THIS_DIR.parent
_WEBAPP_DIR = _BACKEND_DIR.parent
LAB_ROOT = _WEBAPP_DIR.parent  # repo root containing results/, pipeline/, …

# ---------------------------------------------------------------------------
# Shared services (singleton-ish)
# ---------------------------------------------------------------------------

loader = ResultsLoader(LAB_ROOT)
runner = BenchmarkRunner(LAB_ROOT)
runner._results_loader = loader  # allow runner to reload results cache on completion

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Siderust Lab", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev-friendly; lock down for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Wire routers
app.include_router(runs.router, prefix="/api")
app.include_router(experiments.router, prefix="/api")
app.include_router(compare.router, prefix="/api")
app.include_router(benchmark.router, prefix="/api")

# Serve built frontend (production)
_FRONTEND_DIST = _WEBAPP_DIR / "frontend" / "dist"
if _FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")
