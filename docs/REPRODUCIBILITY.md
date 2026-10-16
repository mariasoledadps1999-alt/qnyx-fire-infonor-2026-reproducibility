# Reproducibility Guide & Audit Protocol

This guide outlines the complete protocol to audit, verify, and recompute all scientific experiments presented in the INFONOR 2026 article:
**"Q-NYX Fire: Deterministic Wildfire Propagation Engine, Historical Validation, and Controlled Benchmark Against FlamMap"**.

---

## 1. Prerequisites & Environment Setup

The repository is fully self-contained and designed to run on Windows, Linux, and macOS using Python 3.8 to 3.12.

### Step 1.1: Clone or Enter the Repository
```bash
git clone https://github.com/qnyx-fire/qnyx-fire-infonor-2026-reproducibility.git
cd qnyx-fire-infonor-2026-reproducibility
```

### Step 1.2: Create and Activate an Isolated Virtual Environment
```bash
# On Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

# On Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 1.3: Install Pinned Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Quick Verification (Fast Path)

### Step 2.1: Run Fast Smoke Test (< 2 seconds)
Validates core imports, physical formulas, elliptical eccentricity, and Nelson moisture solver:
```bash
pytest tests/test_smoke.py -v
```

### Step 2.2: Single-Command Verification of All Paper Metrics
Audits data integrity, frozen hashes, CONAF validation metrics, and FlamMap spatial comparison without recomputing:
```bash
python scripts/reproduce_all.py
```

Expected output:
- All data hashes: `[OK] SHA-256 match`
- Talamí 2017 frozen outputs: `[OK] 14,589 cells, 1313.01 ha (All invariants PASS)`
- CONAF Validation: `[OK] IoU=0.0428, Dice=0.0821, CentroidDist=4551.97m`
- FlamMap BENCH-006: 8-case table with exact matching IoU and Dice values.

---

## 3. Individual Experiment Reproduction (Full Path)

### 3.1 Historical Fire Simulation: QNYX-FIRE-TALAMI-001
To recompute the complete 12-hour historical simulation across the 628,480-cell domain:

```bash
# Fast mode (2-hour cropped domain, ~2 seconds):
python scripts/run_talami_simulation.py --mode fast

# Benchmark mode (Full 12-hour domain, ~15-30 seconds):
python scripts/run_talami_simulation.py --mode benchmark
```

Outputs produced in `benchmarks/historical_talami_2017/fire_001_reference/`:
- `TALAMI_2017_FIRE_001_ARRIVAL_TIME.tif` (Minimum arrival time in minutes)
- `TALAMI_2017_FIRE_001_BURNED_MASK.tif` (Binary burned footprint at 12h)
- `TALAMI_2017_FIRE_001_ROS.tif` (Rate of spread field in m/min)
- `TALAMI_2017_FIRE_001_INTENSITY.tif` (Byram fireline intensity in kW/m)
- `TALAMI_2017_FIRE_001_FLAME_LENGTH.tif` (Flame length in meters)
- `TALAMI_2017_FIRE_001_SUMMARY.json` (Diagnostic manifest and hourly evolution table)

### 3.2 Spatial Validation against CONAF: QNYX-VAL-TALAMI-001
To evaluate the simulated 12-hour footprint against the official CONAF historical perimeter:

```bash
python scripts/validate_talami_conaf.py --mode benchmark
```

Outputs produced in `benchmarks/historical_talami_2017/validation_001/`:
- `TALAMI_2017_VAL_001_METRICS.json` (Primary IoU, Dice, centroid shift, directional extents)
- `TALAMI_2017_VAL_001_DIRECTIONAL.csv` (8-sector angular extent table)
- `TALAMI_2017_VAL_001_OVERLAY.tif` (Rasterized confusion mask: TP, FP, FN)
- `TALAMI_2017_VAL_001_OVERVIEW.png` (Spatial overview figure)

### 3.3 Controlled External Benchmark: QNYX-BENCH-006 vs FlamMap
To execute the 8 spatial benchmark cases under wind-slope interaction:

```bash
# Run Q-NYX spatial solver on the 8 benchmark cases (~10 seconds):
python scripts/run_bench_006_flammap.py

# Run spatial comparison against FlamMap 6.2 MTT (~15 seconds):
python scripts/compare_bench_006.py
```

Outputs produced in `results/bench_006/` and `benchmarks/qnyx_bench_006_flammap_wind_slope/analysis/spatial_comparison/`:
- `bench006_spatial_comparison_180min.csv` (Per-case IoU, Dice, area ratios)
- `bench006_arrival_error_common_cells.csv` (Arrival time MAE, RMSE, bias on common cells)
- `bench006_spatial_comparison_audit.json` (Formal validation audit manifest)

---

## 4. Full Recomputation Workflow

To execute all three experimental stages sequentially from scratch:

```bash
python scripts/reproduce_all.py --full-recompute
```

---

## 5. Automated Test Suite Execution

Run the complete scientific test suite:

```bash
pytest tests/ -v
```

Expected result: 136 tests passed (unit tests for Rothermel, directional Albini-Anderson vector combination, Nelson moisture solver, Talamí integration contracts, and BENCH-006 spatial invariants).
