# AGENTS.md

## Purpose
This repository is a benchmarking lab for comparing `siderust` against reference libraries (`erfa`, `astropy`, `libnova`, plus `anise` as available) on accuracy and performance.

## Scope and ownership
- Treat `siderust/`, `anise/`, `astropy/`, `erfa/`, and `libnova/` as vendored submodules unless a task explicitly asks to change them.
- Prefer making changes in:
  - `pipeline/` (experiment orchestration and adapters)
  - `webapp/backend/` (FastAPI API + benchmark runner)
  - `webapp/frontend/` (React/Vite UI)
  - root docs/scripts (`README.md`, `USER_MANUAL.md`, `run.sh`)

## Repository map
- `pipeline/orchestrator.py`: main benchmark entrypoint
- `pipeline/adapters/*`: per-library adapter binaries/scripts
- `results/<YYYY-MM-DD>/<experiment>/`: machine-readable benchmark outputs
- `reports/<YYYY-MM-DD>/<experiment>/`: human-readable reports
- `logs/` and `logs/jobs/`: benchmark job logs and metadata for webapp
- `webapp/backend/app/`: API for runs, compare, and benchmark execution
- `webapp/frontend/`: dashboard UI

## Setup
1. Initialize submodules:
```bash
git submodule update --init --recursive
```
2. Build adapters and Python env:
```bash
./run.sh build
```

## Running benchmarks
- Run everything with defaults (`N=1000`, `seed=42`):
```bash
./run.sh run
```
- Run all with custom size:
```bash
./run.sh run 5000
```
- Run selected experiments directly:
```bash
python3 pipeline/orchestrator.py --experiments gmst_era,kepler_solver --n 2000 --seed 42
```
- Skip performance measurements:
```bash
python3 pipeline/orchestrator.py --experiments all --n 1000 --seed 42 --no-perf
```

Current experiment IDs:
- `frame_rotation_bpn`
- `gmst_era`
- `equ_ecl`
- `equ_horizontal`
- `solar_position`
- `lunar_position`
- `kepler_solver`

## Webapp development
- Backend (from repo root):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r webapp/backend/requirements.txt
uvicorn webapp.backend.app.main:app --reload
```
- Frontend:
```bash
cd webapp/frontend
npm install
npm run dev
```

## Change guidelines
- Keep experiment assumptions explicit (units, time scales, models, reference source).
- Preserve JSON result compatibility for `webapp/backend/app/models/schemas.py`.
- Do not hand-edit generated artifacts in `results/`, `reports/`, or `logs/` unless the task is specifically about fixture/report editing.
- For new experiments, update both orchestration and UI surfaces:
  - input generation + adapter dispatch (`pipeline/orchestrator.py`)
  - API schema expectations (`webapp/backend/app/models/schemas.py`)
  - frontend labels/visualizations (`webapp/frontend/src`)

## Validation checklist
- If adapter/orchestrator logic changed, run at least one targeted experiment and confirm new files under `results/<date>/<experiment>/`.
- If backend changed, confirm API boots and endpoints respond (`/api/runs`, `/api/experiments`).
- If frontend changed, run `npm run build` in `webapp/frontend`.
