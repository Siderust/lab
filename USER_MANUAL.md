# Siderust Lab — User Manual

This repo is a benchmarking + validation “lab” that runs controlled astronomy/astrodynamics experiments and compares **Siderust** against reference libraries (ERFA/SOFA-derived) and peers.

It contains:

- `pipeline/`: input generation, adapters, orchestrator, JSON results
- `results/`: machine-readable outputs written per run date/experiment
- `ui/`: Streamlit GUI + report generator for plots and illustrative reports

## Prerequisites

System tools (recommended on Linux/macOS):

- `git`
- `python3` (and `python3 -m venv`)
- A C compiler + `make` (e.g., `gcc` + `make`)
- Rust toolchain for the Siderust adapter (`cargo`, `rustc`)

## Setup (one-time)

From the repo root:

1) Initialize submodules:

```bash
git submodule update --init --recursive
```

2) Build adapters + create the Python virtualenv:

```bash
./run.sh build
```

This creates/uses a local virtual environment at `.venv/` and installs the Python dependencies needed for the analysis pipeline (`numpy`, `pyerfa`).

## Run the analysis (generate results)

### Run everything (build + run)

```bash
./run.sh
```

By default, this runs all experiments with `N=1000` test cases and `seed=42`.

### Run only the pipeline (already built)

```bash
./run.sh run
```

### Choose the number of cases

```bash
./run.sh run 10000
```

### Run a specific experiment

You can also run the orchestrator directly:

```bash
source .venv/bin/activate
python3 pipeline/orchestrator.py --experiment frame_rotation_bpn --n 5000 --seed 42
python3 pipeline/orchestrator.py --experiment gmst_era --n 5000 --seed 42
python3 pipeline/orchestrator.py --experiment all --n 5000 --seed 42
```

To skip performance measurements (accuracy-only):

```bash
source .venv/bin/activate
python3 pipeline/orchestrator.py --experiment frame_rotation_bpn --n 5000 --seed 42 --no-perf
```

### Where results go

Runs write into:

```text
results/<YYYY-MM-DD>/<experiment>/
  <library>.json
  summary.md
```

Example:

```text
results/2026-02-12/frame_rotation_bpn/siderust.json
results/2026-02-12/frame_rotation_bpn/erfa.json
results/2026-02-12/frame_rotation_bpn/astropy.json
results/2026-02-12/frame_rotation_bpn/summary.md
```

## Run the GUI (plots + interactive exploration)

The GUI is a Streamlit app that reads from `results/` by default.

1) Install GUI dependencies into the same `.venv`:

```bash
source .venv/bin/activate
pip install -r ui/requirements.txt
```

2) Launch the GUI:

```bash
source .venv/bin/activate
streamlit run ui/app.py
```

Then use the left sidebar to select:

- Run date
- Experiment
- Libraries to compare

Tabs include Overview, Accuracy, Performance, Pareto, Outliers, and Reports.

## Generate illustrative reports (headless / CLI)

The report generator reads a single `results/<date>/<experiment>/` folder and writes to `reports/`:

```bash
source .venv/bin/activate
pip install -r ui/requirements.txt

python -m ui.report_generator \
  --results-dir results \
  --date 2026-02-12 \
  --experiment frame_rotation_bpn \
  --output-dir reports \
  --format both
```

Outputs:

```text
reports/<date>/<experiment>/index.html
reports/<date>/<experiment>/report.md
```

## Troubleshooting

### “No results found” in the GUI

Run the pipeline at least once:

```bash
./run.sh run 1000
```

### Missing `streamlit` / `plotly` / `pandas`

Install GUI deps in your active venv:

```bash
source .venv/bin/activate
pip install -r ui/requirements.txt
```

### Report HTML has no embedded PNG plots

The HTML report embeds static PNGs when Plotly image export works. Install `kaleido` (included in `ui/requirements.txt`) and rerun the report generator:

```bash
source .venv/bin/activate
pip install -r ui/requirements.txt
python -m ui.report_generator --results-dir results --date <date> --experiment <exp> --format html
```

### Run UI sanity tests

```bash
source .venv/bin/activate
pip install pytest
python -m pytest ui/test_loader.py -v
```

