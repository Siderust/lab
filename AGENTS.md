# AGENTS.md

## Purpose
This repository is a virtual laboratory for measuring `siderust` against other astronomy and astrodynamics tools on two axes:

- accuracy against explicit reference sources
- performance under reproducible workloads

The lab exists to answer comparative questions with traceable assumptions, not to present any single implementation as authoritative by itself.

## Normative references
Use these as the long-term reference policy for the repo:

- **Transformations and time/earth-orientation work**: treat **SOFA / IAU conventions** as the normative reference. In practice the lab may use ERFA- or Astropy-backed adapters to execute those models, but the authority is the underlying SOFA model and its documented conventions.
- **Ephemerides for Sun, Moon, and planets**: treat **JPL ephemerides** as the normative reference. When the pipeline fetches external truth data, prefer JPL Horizons or a documented JPL DE source and record the exact source tag when available.

If a benchmark cannot match the reference model exactly across all tools, document the mismatch explicitly instead of hiding it inside aggregate metrics.

## Repository ownership
Assume these directories are vendored third-party code unless a task explicitly asks to modify them:

- `siderust/`
- `anise/`
- `astropy/`
- `erfa/`
- `libnova/`

Prefer making lab changes in these first-party areas:

- `pipeline/` for experiment orchestration, adapters, input generation, and reference-data plumbing
- `webapp/backend/` for the FastAPI API and benchmark execution services
- `webapp/frontend/` for the React/Vite dashboard
- root docs and scripts such as `README.md`, `USER_MANUAL.md`, and `run.sh`

## Stable repo map
These paths are the main long-lived entrypoints:

- `pipeline/orchestrator.py`: benchmark orchestration entrypoint
- `pipeline/adapters/`: per-tool adapter implementations
- `pipeline/horizons_client.py`: JPL Horizons integration and caching
- `pipeline/tests/`: benchmark and reference-data tests
- `webapp/backend/app/main.py`: API entrypoint
- `webapp/backend/app/models/schemas.py`: result and API schema contracts
- `webapp/frontend/src/`: UI pages, charts, and API client code
- `results/`, `latest_results/`, and `logs/`: generated run outputs and job metadata

Do not hard-code assumptions in this file about the current list of experiments, exact benchmark modes, or the current set of adapters. Those can evolve.

## Working rules
- Keep experiment assumptions explicit: units, frames, time scales, EOP inputs, geodesy, refraction, aberration/light-time policy, and ephemeris source.
- Preserve result-schema compatibility unless the task explicitly includes a schema migration.
- Treat generated outputs under `results/`, `latest_results/`, `reports/`, and `logs/` as artifacts, not hand-edited source files, unless the task is specifically about fixtures or reports.
- When adding or changing an experiment, update every affected surface together: orchestration, adapters, schemas, and UI.
- Prefer naming references by **model/source** (`SOFA`, `JPL Horizons`, `DE441`, etc.) rather than by a wrapper implementation alone.
- Record provenance whenever possible: library version, git SHA, kernel/source tag, seed, sample count, and machine/runtime metadata.

## Validation
- If `pipeline/` changes, run at least one targeted benchmark or the relevant tests.
- If Horizons or other reference-fetch logic changes, run the related tests in `pipeline/tests/`.
- If backend code changes, verify the API starts and key endpoints still respond.
- If frontend code changes, run `npm run build` in `webapp/frontend`.

## What to avoid
- Do not redefine a candidate library as the ground truth just because it is convenient to call from the pipeline.
- Do not compare tools without stating model parity limits.
- Do not edit vendored submodules for lab behavior changes when the same change belongs in adapters or orchestration.
