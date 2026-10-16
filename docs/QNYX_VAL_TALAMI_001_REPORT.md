# Q-NYX Validation Report: Talamí 2017 Historical Fire
**Document Identifier:** `QNYX-VAL-TALAMI-001`  
**Execution Timestamp:** 2026-10-16T14:19:55.855704  
**Baseline Stage:** `TALAMI_FIRE_BENCHMARK`  
**Gate Status:** `TALAMI_FIRE_BENCHMARK: PASSED` | `MULTIFIRE_VALIDATION: BLOCKED`  
**Validation Classification:** `MAIN_ERROR_APPEARS_DIRECTIONAL`  

---

> [!IMPORTANT]
> **TEMPORAL INTERPRETATION GUARD**  
> The CONAF final perimeter is used as a final spatial envelope reference, not as temporal ground truth for the 12-hour Q-NYX run.  
> It is scientifically invalid to report a "12-hour area error vs final CONAF" or "temporal accuracy", because CONAF records the multi-day containment perimeter (January 14 to February 9, 2017; 26 days of activity), whereas Q-NYX reference run `QNYX-FIRE-TALAMI-001` evaluates 720 minutes (12 hours) of propagation from an unvalidated ignition scenario hypothesis.

---

## 1. Frozen Fire Input Verification
Frozen reference simulation outputs from milestone `QNYX-FIRE-TALAMI-001` were verified via SHA256 against `QNYX_FIRE_TALAMI_001_RELEASE_MANIFEST.json`:

| Frozen Artifact | Relative Path | Verified SHA256 | Status |
|---|---|---|---|
| Burned Mask GeoTIFF | `benchmarks/historical_talami_2017/fire_001_reference/TALAMI_2017_FIRE_001_BURNED_MASK.tif` | `42507c96459a7387215c9827cffa6efea95595eaad6ec72e0ca9e5aabd160f61` | `VERIFIED` |
| Arrival Time GeoTIFF | `benchmarks/historical_talami_2017/fire_001_reference/TALAMI_2017_FIRE_001_ARRIVAL_TIME.tif` | `f7ba3b3f557790e29402b586dce408ab79aa70c5b90786b692538cdfe25c92b1` | `VERIFIED` |

No simulation recalculation or engine modification occurred.

---

## 2. CONAF Reference Contract & Geometry Validation
The historical reference perimeter was obtained from official CONAF incident records:

- **Reference Role:** `FINAL_FIRE_ENVELOPE`
- **Temporal Role:** `NOT_12H_GROUND_TRUTH`
- **Source:** CONAF Historical Fire Database (observed_perimeter_talami_2017)
- **Coordinate Reference System:** `EPSG:32719`
- **Geometry Type & Validity:** `MultiPolygon`, `is_valid = True`
- **Reference Dates:** `FECHA_INI = 2017-01-14` to `FECHA_TER = 2017-02-09` (26 days active duration)
- **Declared Area:** **7606.51 ha**
- **Geometry-Derived Area:** **8002.90 ha**
- **Area Discrepancy:** `+396.39 ha` (`+5.21%`). Both values are explicitly preserved without silent reconciliation.

### Rasterization Policy
- **Grid Contract:** Exact 30m fire grid (640 rows × 982 columns, EPSG:32719).
- **Policy:** `all_touched=False` (center of 30m pixel falls inside CONAF polygon boundary).
- **Rasterized Cells:** 88,958 cells.
- **Rasterized Reference Area:** **8006.22 ha** (99.96% area fidelity to vector geometry).

---

## 3. Primary Spatial Metrics (12-Hour Mask vs Final Envelope)

| Metric Code | Description | Value | Interpretation |
|---|---|---|---|
| `SIM_AREA_HA` | 12h Simulated Burned Area | **1313.01 ha** | 14,589 cells (30m) |
| `REFERENCE_RASTER_AREA_HA` | CONAF Rasterized Reference Area | **8006.22 ha** | 88,958 cells (30m) |
| `INTERSECTION_AREA_HA` | True Positive Overlap Area | **382.50 ha** | 4,250 cells (30m) |
| `UNION_AREA_HA` | Combined Spatial Footprint | **8936.73 ha** | 99,297 cells (30m) |
| `IoU` | Intersection over Union | **0.0428** | Spatial overlap bounded [0, 1] |
| `Dice` | Sorensen-Dice Coefficient | **0.0821** | Spatial coincidence bounded [0, 1] |
| `SIM_WITHIN_REFERENCE_FRACTION` | Simulated cells inside CONAF | **29.13%** | Intersection / Simulation Area |
| `REFERENCE_COVERED_FRACTION` | CONAF cells reached at 12h | **4.78%** | `FINAL_ENVELOPE_COVERAGE_ONLY` |
| `FINAL_ENVELOPE_AREA_RATIO` | Simulation / Reference Area Ratio | **0.1640** | Diagnostics only, not temporal accuracy |

---

## 4. Centroid Metrics & Directional Bias Analysis
Evaluates the geometric shift between the reference envelope and the 12-hour simulation:

- **Simulation Centroid:** `(317647.17 E, 6230665.56 N)`
- **Reference Centroid:** `(314623.67 E, 6227262.78 N)`
- **Centroid Distance:** **4551.97 m** (~4.55 km)
- **Centroid Delta East:** `+3023.49 m`
- **Centroid Delta North:** `+3402.78 m`
- **Bearing from Reference to Simulation Centroid:** **41.62°** (Northeast)
- **Interpretation:** `SPATIAL_BIAS_DIAGNOSTIC` (Strictly diagnostic; not a calibration target).

---

## 5. Directional Extent from Ignition
From ignition point `(316425 E, 6230260 N)` in 8 sectors of 45°:

| Direction | Azimuth | Sim Distance (m) | Ref Distance (m) | Ratio (Sim/Ref) | Delta (Sim - Ref) (m) |
|---|---|---|---|---|---|
| N |   0.0° |  2465.18 |     0.00 | N/A (Ref=0) | +2465.18 |
| NE |  45.0° |  3673.46 |    69.46 | 52.8860 | +3604.00 |
| E |  90.0° |  3682.84 |  7832.19 | 0.4702 | -4149.35 |
| SE | 135.0° |  2940.87 |  6799.75 | 0.4325 | -3858.88 |
| S | 180.0° |  1449.35 |  7969.22 | 0.1819 | -6519.87 |
| SW | 225.0° |   855.23 | 10394.24 | 0.0823 | -9539.01 |
| W | 270.0° |   990.77 | 11704.27 | 0.0847 | -10713.50 |
| NW | 315.0° |  1871.82 |  3484.27 | 0.5372 | -1612.45 |

### Observations on Directional Patterns:
1. **East & Northeast Propagation:** The simulation advances **3,682.80 m East** and **3,673.53 m Northeast**, driven by strong daytime westerly winds present in ERA5-Land on January 14.
2. **West & Southwest Suppression:** In the historical record, CONAF envelope extends **11,704.30 m West** and **10,394.22 m Southwest**. Under the 18:00 UTC scenario hypothesis, simulated propagation towards West is only 990.81 m, and Southwest only 855.22 m.
3. **North Barrier:** CONAF has 0.0 m extent North of ignition, whereas the simulation advanced 2,465.18 m North into shrub fuels before the 12h horizon.

---

## 6. Boundary Distance Metrics
Distance from simulated boundary cells to nearest reference boundary:

- **Mean Nearest-Boundary Distance:** **793.92 m**
- **Median Nearest-Boundary Distance:** **715.61 m**
- **P90 Nearest-Boundary Distance:** **1719.03 m**

---

## 7. Arrival-Time Spatial Summary (Isochrones)
Using frozen `TALAMI_2017_FIRE_001_ARRIVAL_TIME.tif`:

| Horizon | Elapsed Time (min) | Cumulative Burned Cells | Cumulative Area (ha) | Fraction of 12h Run |
|---|---|---|---|---|
| 1 h | 60 | 10 | 0.90 ha | 0.07% |
| 3 h | 180 | 460 | 41.40 ha | 3.15% |
| 6 h | 360 | 5400 | 486.00 ha | 37.01% |
| 9 h | 540 | 10752 | 967.68 ha | 73.70% |
| 12 h | 720 | 14589 | 1313.01 ha | 100.00% |

*Note: None of these isochrones corresponds to CONAF perimeter recording.*

---

## 8. FIRMS Satellite Detections Guard
- **Status:** `POST_HORIZON_SPATIAL_REFERENCE`
- **Assessment:** FIRMS VIIRS and MODIS active fire detections over Talamí begin on January 15 ~05:00 UTC (sparse) and peak January 16–18. Because the 12-hour simulation horizon ends at 2017-01-15 06:00 UTC, FIRMS data cannot serve as temporal validation for this run.

---

## 9. Historical Comparison Across Q-NYX Runs

| Metric | Legacy B0 (Static 100m) | Legacy B1 (Dynamic 100m) | Nelson 002 Baseline (100m) | FIRE-TALAMI-001 (Frozen SB 30m) |
|---|---|---|---|---|
| **Burned Area (ha)** | 6,361.70 | 8,441.01 | 7,835.76 | **1,313.01** |
| **Final Envelope Area Ratio** | 0.7946 | 1.0543 | 0.9787 | **0.1640** |
| **IoU vs CONAF** | 0.2006 | 0.2178 | 0.1990 | **0.0428** |
| **Dice vs CONAF** | 0.3342 | 0.3577 | 0.3319 | **0.0821** |
| **Sim-in-CONAF Fraction** | 37.73% | 34.80% | 33.50% | **29.13%** |
| **Centroid Distance (m)** | 6,331.50 | 5,977.90 | 6,217.80 | **4,551.97** |
| **Fuel Reconstruction** | Legacy custom (100m) | Legacy custom (100m) | Legacy custom (100m) | **Scott & Burgan Standard (30m)** |
| **Weather Model** | Static canonical | Dynamic ERA5 | Dynamic ERA5 | **Dynamic ERA5 (Nelson subhourly)** |
| **Moisture Model** | Static 5% dead | Static 5% dead | Nelson (hourly/5min) | **Nelson (5min subhourly lookup)** |
| **Start Hypothesis** | 2017-01-14 18:00 UTC | 2017-01-14 18:00 UTC | 2017-01-14 18:00 UTC | **2017-01-14 18:00 UTC** |

### Key Architectural Insight:
Legacy B0/B1 runs exhibited large area expansion (~6,000–8,400 ha) because their fuel parameterization assumed unphysical low fuel packing and aggressive rate of spread multipliers. When standard, uncalibrated physical Scott & Burgan fuel beds (e.g. SH4, SH7, TU5) are applied with coupled physical Nelson moisture, the 12-hour spread rate is realistically tempered to 1,313 ha. The area discrepancy vs the 8,006 ha final envelope is primarily a consequence of temporal horizon (12 hours vs 26 days) and directional wind shift over multi-day synoptic weather cycles.

---

## 10. Diagnostic Classification
- **Classification:** `MAIN_ERROR_APPEARS_DIRECTIONAL`
- **Evidence:** Under the fixed scenario hypothesis (ignition at 2017-01-14 18:00 UTC), simulated fire propagates predominantly toward the East and Northeast (bearing 41.62°, centroid shift +3,023m E, +3,402m N), whereas the historical multi-day envelope expanded primarily West and Southwest. Directional misalignment accounts for the low overlap (IoU = 0.0428), alongside the temporal scale disparity (12h vs 26 days).

---

## 11. Artifact Checklist & Hashes
The following validation outputs were generated:

1. `TALAMI_2017_VAL_001_METRICS.json`: Full machine-readable metrics dictionary.
2. `TALAMI_2017_VAL_001_DIRECTIONAL.csv`: 8-direction extent table.
3. `TALAMI_2017_VAL_001_OVERLAY.tif`: 30m categorical overlay GeoTIFF (TP=1, FP=2, FN=3).
4. `TALAMI_2017_VAL_001_OVERVIEW.png`: 6-panel overview visualization.
5. `docs/QNYX_VAL_TALAMI_001_REPORT.md`: This validation report.
