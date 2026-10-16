#!/usr/bin/env python3
"""
scripts/reproduce_all.py

Single-command master reproduction and auditing script for the INFONOR 2026 paper:
"Q-NYX Fire: Deterministic Wildfire Propagation Engine, Historical Validation,
and Controlled Benchmark Against FlamMap".

Workflows supported:
1. Default / Verification Mode (`python scripts/reproduce_all.py`):
   - Fast audit of scientific data integrity, frozen manifests, release outputs,
     and reported metrics in seconds.
2. Full Recomputation Mode (`python scripts/reproduce_all.py --full-recompute`):
   - Re-runs the complete 12h Talamí simulation, evaluates CONAF validation,
     executes the 8 Q-NYX spatial cases of BENCH-006, and recomputes the
     FlamMap spatial comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def run_step(step_name: str, cmd: list[str]) -> bool:
    print(f"\n{'='*70}")
    print(f"=== STEP: {step_name} ===")
    print(f"{'='*70}")
    t0 = time.time()
    res = subprocess.run([sys.executable] + cmd, cwd=REPO_ROOT)
    duration = time.time() - t0
    if res.returncode == 0:
        print(f"[PASSED] {step_name} ({duration:.2f} s)")
        return True
    else:
        print(f"[FAILED] {step_name} returned exit code {res.returncode}")
        return False


def print_paper_metrics_summary():
    print(f"\n{'='*70}")
    print("=== SUMMARY OF NUMERICAL RESULTS REPORTED IN INFONOR 2026 PAPER ===")
    print(f"{'='*70}")

    # 1. Talamí 2017 Simulation
    summary_file = REPO_ROOT / "benchmarks" / "historical_talami_2017" / "fire_001_reference" / "TALAMI_2017_FIRE_001_SUMMARY.json"
    if summary_file.exists():
        with open(summary_file, "r", encoding="utf-8") as f:
            t_sum = json.load(f)
        b_res = t_sum.get("burned_results", {})
        invs = t_sum.get("integration_invariants", {})
        print("\n[1] QNYX-FIRE-TALAMI-001 (12h Simulation Hypothesis):")
        print(f"    - Domain: 640 x 982 cells (56,563.20 ha) at 30m resolution")
        print(f"    - Burned Cells at 12h: {b_res.get('burned_cells')} cells")
        print(f"    - Burned Area at 12h: {b_res.get('burned_area_ha'):.2f} ha")
        print(f"    - Mean Rate of Spread: {b_res.get('mean_ros_m_min', 0):.2f} m/min")
        print(f"    - 95th Percentile ROS: {b_res.get('p95_ros_m_min', 0):.2f} m/min")
        print(f"    - Maximum ROS: {b_res.get('max_ros_m_min', 0):.2f} m/min")
        print(f"    - Mean Fireline Intensity: {b_res.get('mean_intensity_kw_m', 0):.2f} kW/m")
        print(f"    - Mean Flame Length: {b_res.get('mean_flame_length_m', 0):.2f} m")
        print(f"    - Invariants Verified: {sum(invs.values())}/{len(invs)} PASS")

    # 2. CONAF Validation
    val_file = REPO_ROOT / "benchmarks" / "historical_talami_2017" / "validation_001" / "TALAMI_2017_VAL_001_METRICS.json"
    if val_file.exists():
        with open(val_file, "r", encoding="utf-8") as f:
            v_sum = json.load(f)
        prim = v_sum.get("primary_metrics", {})
        cent = v_sum.get("centroid_metrics", {})
        print("\n[2] QNYX-VAL-TALAMI-001 (Validation against CONAF Final Perimeter):")
        print(f"    - CONAF Rasterized Reference Area: {prim.get('reference_raster_area_ha'):.2f} ha (88,958 cells)")
        print(f"    - 12h Simulated Burned Area: {prim.get('sim_area_ha'):.2f} ha (14,589 cells)")
        print(f"    - Intersection (True Positive): {prim.get('intersection_area_ha'):.2f} ha (4,250 cells)")
        print(f"    - Intersection over Union (IoU): {prim.get('iou'):.4f}")
        print(f"    - Sorensen-Dice Coefficient: {prim.get('dice'):.4f}")
        print(f"    - Fraction within Reference: {prim.get('sim_within_reference_fraction', 0)*100:.2f}%")
        print(f"    - Centroid Distance Shift: {cent.get('centroid_distance_m'):.2f} m (Bearing {cent.get('bearing_reference_to_simulation_deg'):.2f}° {cent.get('bearing_cardinal')})")
        print("    - Scientific Guard: CONAF records 26-day containment envelope; not temporal ground truth.")

    # 3. BENCH-006 Comparison vs FlamMap
    bench_file = REPO_ROOT / "results" / "bench_006" / "bench006_spatial_comparison_180min.csv"
    if bench_file.exists():
        import csv
        print("\n[3] QNYX-BENCH-006 (Controlled Spatial Comparison vs FlamMap 6.2 MTT at 180 min):")
        print("    " + "-"*65)
        print("    Case | Role                       | Q-NYX (ha) | FlamMap (ha) | IoU   | Dice")
        print("    " + "-"*65)
        with open(bench_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                c_num = r.get("case_num", "")
                role = r.get("role", "")[:26]
                q_ha = float(r.get("qnyx_area_ha", 0))
                f_ha = float(r.get("flammap_area_ha", 0))
                iou = float(r.get("iou", 0))
                dice = float(r.get("dice", 0))
                print(f"    {c_num:4} | {role:26} | {q_ha:10.2f} | {f_ha:12.2f} | {iou:.3f} | {dice:.3f}")
        print("    " + "-"*65)


def main():
    parser = argparse.ArgumentParser(description="INFONOR 2026 Reproduction Master Script")
    parser.add_argument("--full-recompute", action="store_true",
                        help="Re-runs all simulations and spatial comparisons from scratch.")
    parser.add_argument("--verify-only", action="store_true", default=False,
                        help="Verifies all artifact hashes, configs, and metrics without recomputation (default).")
    args = parser.parse_args()

    full_recompute = args.full_recompute

    print(f"{'='*70}")
    print("=== Q-NYX FIRE: INFONOR 2026 REPRODUCIBILITY SUITE ===")
    print(f"Mode: {'FULL RECOMPUTATION' if full_recompute else 'VERIFICATION & AUDITING'}")
    print(f"Repository Root: {REPO_ROOT}")
    print(f"{'='*70}")

    steps = []

    # Step 1: External data integrity audit
    steps.append(("External Data Integrity Check", ["scripts/download_external_data.py", "--verify"]))

    if full_recompute:
        # Step 2: Run Talamí simulation
        steps.append(("Talamí 2017 Simulation Run", ["scripts/run_talami_simulation.py", "--mode", "benchmark"]))
        # Step 3: Run CONAF validation
        steps.append(("Talamí CONAF Validation", ["scripts/validate_talami_conaf.py", "--mode", "benchmark"]))
        # Step 4: Run BENCH-006 cases
        steps.append(("BENCH-006 Spatial Simulation", ["scripts/run_bench_006_flammap.py"]))
        # Step 5: Run FlamMap comparison
        steps.append(("BENCH-006 FlamMap Spatial Comparison", ["scripts/compare_bench_006.py"]))
    else:
        # Fast verification of frozen artifacts
        steps.append(("Talamí 2017 Frozen Output Verification", ["scripts/run_talami_simulation.py", "--mode", "verify"]))
        steps.append(("Talamí CONAF Validation Verification", ["scripts/validate_talami_conaf.py", "--mode", "verify"]))
        steps.append(("BENCH-006 Comparison Script Audit", ["scripts/compare_bench_006.py"]))

    # Step N: Run pytest smoke test suite
    steps.append(("Smoke Tests Execution", ["-m", "pytest", "tests/test_smoke.py", "-v"]))

    all_passed = True
    for name, cmd in steps:
        ok = run_step(name, cmd)
        if not ok:
            all_passed = False
            print(f"\n[ABORT] Pipeline stopped at failed step: {name}")
            sys.exit(1)

    print_paper_metrics_summary()

    print(f"\n{'='*70}")
    print("[SUCCESS] ALL REPRODUCIBILITY CHECKS AND AUDITS PASSED PERFECTLY!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
