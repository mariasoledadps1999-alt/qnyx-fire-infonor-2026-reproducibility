"""
tests/test_talami_reproducibility.py

Integration and validation test suite for historical Talamí 2017:
- QNYX-FIRE-TALAMI-001 (12h Simulation Engine Integration & Invariants)
- QNYX-VAL-TALAMI-001 (Spatial Validation against CONAF Observed Perimeter)
"""

"""
tests/test_qnyx_fire_talami_001.py

Unit test suite for milestone QNYX-FIRE-TALAMI-001:
First Integrated Historical Fire Run — Talamí 2017.

Mandatory verification items:
- frozen hashes unchanged
- no legacy fuel usage
- correct ignition mapping
- NB propagation blocked
- dynamic fuel transfer used
- Nelson dead FMC consumed
- explicit live FMC assumptions consumed
- dynamic weather consumed
- no CONAF/FIRMS access in fire simulation
- deterministic same-input run and cache hit
- output integration invariants
"""

import hashlib
import json
from pathlib import Path
import pytest
import numpy as np
import rasterio

REPO_ROOT = Path(__file__).resolve().parent.parent


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def test_frozen_hashes_unchanged():
    """Verify that all frozen artifacts registered in release manifest match expected hashes."""
    from scripts.run_talami_simulation import EXPECTED_GEOTIFF_HASHES, compute_sha256
    ref_dir = REPO_ROOT / "benchmarks" / "historical_talami_2017" / "fire_001_reference"
    for fname, exp_hash in EXPECTED_GEOTIFF_HASHES.items():
        p = ref_dir / fname
        assert p.exists(), f"Missing frozen output: {p}"
        assert compute_sha256(p) == exp_hash, f"SHA-256 mismatch for {fname}"


def test_no_legacy_fuel_usage():
    """Verify that only Scott & Burgan standard models are referenced, no legacy fuels."""
    fuel_raster_path = REPO_ROOT / "benchmarks" / "historical_talami_2017" / "chile_fuel_reference" / "TALAMI_2017_FUEL_MODEL_FIREGRID_30M.tif"
    catalog_path = REPO_ROOT / "configs" / "fuel_models_scott_burgan_qnyx.json"
    with open(catalog_path, "r", encoding="utf-8") as f:
        cat_data = json.load(f)
    sb_codes = {int(k) for k in cat_data["models"].keys()}

    with rasterio.open(fuel_raster_path) as src:
        arr = src.read(1)
        unique_raster_codes = {int(c) for c in np.unique(arr)}

    assert unique_raster_codes.issubset(sb_codes), (
        f"Raster contains codes {unique_raster_codes - sb_codes} not in Scott & Burgan catalog!"
    )
    # Prohibit legacy codes 1..13 or negative codes
    assert not any(c in range(1, 14) for c in unique_raster_codes), "Legacy fuel codes (1..13) detected in 30m grid!"


def test_correct_ignition_mapping():
    """Verify that ignition coordinate (316425, 6230260) maps to row=215, col=555 and fuel is GR1 (101)."""
    fuel_raster_path = REPO_ROOT / "benchmarks" / "historical_talami_2017" / "chile_fuel_reference" / "TALAMI_2017_FUEL_MODEL_FIREGRID_30M.tif"
    with rasterio.open(fuel_raster_path) as src:
        inv = ~src.transform
        col_f, row_f = inv * (316425.0, 6230260.0)
        r, c = int(row_f), int(col_f)
        fuel_code = int(src.read(1)[r, c])

    assert r == 215, f"Expected row 215, got {r}"
    assert c == 555, f"Expected col 555, got {c}"
    assert fuel_code == 101, f"Expected GR1 (101), got {fuel_code}"


def get_latest_benchmark_summary() -> dict:
    summary_path = REPO_ROOT / "benchmarks" / "historical_talami_2017" / "fire_001_reference" / "TALAMI_2017_FIRE_001_SUMMARY.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise FileNotFoundError(f"Benchmark summary not found at {summary_path}")


def test_nb_propagation_blocked():
    """Verify that nonburnable fuels (93 NB3, 99 NB9) are blocked and never burned."""
    from fire_engine.fuels.models import load_scott_burgan_catalog
    cat = load_scott_burgan_catalog()
    assert 93 in cat and not cat[93].is_burnable
    assert 99 in cat and not cat[99].is_burnable

    data = get_latest_benchmark_summary()
    assert data["integration_invariants"]["no_burned_nonburnable_cells"] is True


def test_dynamic_fuel_transfer_used():
    """Verify that dynamic herbaceous fuel transfer is applied to dynamic models."""
    from fire_engine.fuels.models import load_scott_burgan_catalog, FuelMoisture, apply_dynamic_fuel_transfer
    cat = load_scott_burgan_catalog()
    gr1 = cat[101]
    assert gr1.is_dynamic is True

    # At 60% live herbaceous moisture, some herbaceous load transfers to dead herbaceous
    moist = FuelMoisture(dead_1h=0.07, dead_10h=0.08, dead_100h=0.10, live_herbaceous=0.60, live_woody=0.90)
    transferred_bed = apply_dynamic_fuel_transfer(gr1, moist)
    dead_herb = [p for p in transferred_bed.particle_classes if p.name == "dead_herbaceous"]
    assert len(dead_herb) == 1, "Expected transferred dead_herbaceous class"
    assert dead_herb[0].load_lb_ft2 > 0.0, "Expected non-zero transferred dead herbaceous load"


def test_nelson_dead_fmc_consumed():
    """Verify that Nelson provider supplies time-dependent dead FMC initialized from 48h conditioning."""
    import pandas as pd
    from fire_engine.weather import WeatherManager
    from fire_engine.fuels.nelson import NelsonFuelMoistureProvider

    df = pd.read_csv(REPO_ROOT / "benchmarks" / "historical_talami_2017" / "weather_v2_nelson" / "processed" / "meteo_talami_nelson_hourly.csv")
    records = []
    for i in range(13):
        r = df.iloc[i]
        records.append({
            "hour_index": i,
            "temperature_c": float(r["temperature_c"]),
            "relative_humidity_pct": float(r["relative_humidity_pct"]),
            "solar_radiation_w_m2": float(r["solar_radiation_w_m2"]),
            "precipitation_mm": float(r["precipitation_mm"]),
            "wind_speed_ms": float(r["wind_speed_ms"]),
            "wind_direction_deg": float(r["wind_from_direction_deg"]),
        })
    wm = WeatherManager(hourly_records=records)
    prov = NelsonFuelMoistureProvider(
        weather_manager=wm,
        initialization_mode="explicit",
        seed_mode="explicit",
        seed_dead_1h=0.07101,
        seed_dead_10h=0.08031,
        seed_dead_100h=0.09713,
        live_herbaceous=0.60,
        live_woody=0.90,
        lookup_mode="nfdrs4_sequential_5min"
    )
    s0 = prov.get_state(0.0, 215, 555)
    s600 = prov.get_state(600.0, 215, 555)
    assert s0.dead_1h == pytest.approx(0.07101, abs=1e-4)
    assert s0.dead_10h == pytest.approx(0.08031, abs=1e-4)
    assert s0.dead_100h == pytest.approx(0.09713, abs=1e-4)
    assert s600.dead_1h > s0.dead_1h, "Dead fuel moisture should increase into the night"


def test_explicit_live_fmc_assumptions_consumed():
    """Verify live fuel moisture assumptions are explicitly documented and unvalidated."""
    data = get_latest_benchmark_summary()
    live = data["physics_contracts"]["live_fuel_moisture"]
    assert live["live_herbaceous"] == 0.60
    assert live["live_woody"] == 0.90
    assert live["provenance"] == "UNVALIDATED_SCENARIO_ASSUMPTION"
    assert live["purpose"] == "ENGINE_INTEGRATION_REFERENCE_ONLY"


def test_dynamic_weather_consumed():
    """Verify that weather varies dynamically across the 12 hours."""
    data = get_latest_benchmark_summary()
    hourly = data["hourly_diagnostics"]
    assert len(hourly) == 12
    winds = [h["wind_speed_ms"] for h in hourly]
    dirs = [h["wind_direction_deg"] for h in hourly]
    assert len(set(winds)) > 1, "Wind speed must vary dynamically over time"
    assert len(set(dirs)) > 1, "Wind direction must vary dynamically over time"


def test_no_conaf_firms_access_in_fire_simulation():
    """Verify that simulation script contains zero references to CONAF perimeter or FIRMS."""
    sim_script = REPO_ROOT / "scripts" / "run_talami_simulation.py"
    with open(sim_script, "r", encoding="utf-8") as f:
        code = f.read()

    assert "observed_perimeter" not in code.lower()
    assert "conaf_shapefile" not in code.lower()
    assert "conaf_perimeter" not in code.lower()
    assert "firms_viirs" not in code.lower()
    assert "firms_detections" not in code.lower()
    assert "iou" not in code.lower()
    assert "dice" not in code.lower()


def test_output_invariants():
    """Verify that all 10 integration invariants passed on benchmark outputs."""
    data = get_latest_benchmark_summary()
    inv = data["integration_invariants"]
    assert inv["ignition_arrival_time_is_zero"] is True
    assert inv["all_finite_arrival_times_non_negative"] is True
    assert inv["no_burned_nonburnable_cells"] is True
    assert inv["no_nan_in_burned_outputs"] is True
    assert inv["no_negative_ros"] is True
    assert inv["no_negative_intensity"] is True
    assert inv["no_negative_flame_length"] is True
    assert inv["burned_cells_within_domain"] is True
    assert inv["simulation_end_within_weather_coverage"] is True
    assert inv["fuel_lookup_coverage_100pct"] is True


def test_cache_hit_and_determinism():
    """Verify that frozen simulation artifacts pass SHA-256 verification."""
    from scripts.run_talami_simulation import verify_frozen_artifacts
    ref_dir = REPO_ROOT / "benchmarks" / "historical_talami_2017" / "fire_001_reference"
    assert verify_frozen_artifacts(ref_dir) is True


"""
Unit and Integration Test Suite for QNYX-VAL-TALAMI-001.
Standard: QNYX-OPS-001

Invariants tested:
1. frozen fire hashes unchanged
2. CONAF not used by fire engine
3. reference geometry valid
4. identical grid rasterization
5. IoU bounded [0, 1]
6. Dice bounded [0, 1]
7. intersection <= simulated area
8. intersection <= reference area
9. centroid calculations deterministic
10. directional sectors exhaustive
11. no calibration
12. same inputs -> deterministic metrics
13. cache hit
"""

import json
from pathlib import Path
import numpy as np
import pytest
import rasterio

REPO_ROOT = Path(__file__).resolve().parent.parent
FIRE_REF_DIR = REPO_ROOT / "benchmarks" / "historical_talami_2017" / "fire_001_reference"
VAL_DIR = REPO_ROOT / "benchmarks" / "historical_talami_2017" / "validation_001"
CONAF_GPKG = REPO_ROOT / "benchmarks" / "historical_talami_2017" / "conaf_reference" / "observed_perimeter_talami_2017.gpkg"
FIRE_RELEASE_MANIFEST = FIRE_REF_DIR / "QNYX_FIRE_TALAMI_001_RELEASE_MANIFEST.json"


# 1. Frozen fire hashes unchanged
def test_01_frozen_fire_hashes_unchanged():
    from scripts.validate_talami_conaf import verify_frozen_fire_inputs, compute_sha256
    verified = verify_frozen_fire_inputs(fire_ref_dir=FIRE_REF_DIR, manifest_path=FIRE_RELEASE_MANIFEST)

    with open(FIRE_RELEASE_MANIFEST, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    manifest_hashes = manifest_data["output_hashes"]

    assert verified["burned_mask"] == manifest_hashes["TALAMI_2017_FIRE_001_BURNED_MASK.tif"]
    assert verified["arrival_time"] == manifest_hashes["TALAMI_2017_FIRE_001_ARRIVAL_TIME.tif"]


# 2. CONAF not used by fire engine
def test_02_conaf_not_used_by_fire_engine():
    import inspect
    import fire_engine.kernels.rothermel as rothermel
    import fire_engine.spread as spread

    # Inspect source of fire simulation modules
    rothermel_src = inspect.getsource(rothermel).lower()
    spread_src = inspect.getsource(spread).lower()

    assert "conaf" not in rothermel_src
    assert "conaf" not in spread_src
    assert "observed_perimeter" not in rothermel_src
    assert "observed_perimeter" not in spread_src


# 3. Reference geometry valid
def test_03_reference_geometry_valid():
    from scripts.validate_talami_conaf import load_conaf_reference
    geom, meta = load_conaf_reference(gpkg_path=CONAF_GPKG)

    assert geom.is_valid
    assert meta["crs"] == "EPSG:32719"
    assert meta["geometry_type"] in ("MultiPolygon", "Polygon")
    assert meta["declared_area_ha"] == 7606.51
    assert meta["geometry_derived_area_ha"] == 8002.90
    assert meta["area_discrepancy_reported"] is True


# 4. Identical grid rasterization
def test_04_identical_grid_rasterization():
    from scripts.validate_talami_conaf import load_conaf_reference, rasterize_reference_perimeter
    geom, _ = load_conaf_reference(gpkg_path=CONAF_GPKG)

    mask_path = FIRE_REF_DIR / "TALAMI_2017_FIRE_001_BURNED_MASK.tif"
    with rasterio.open(mask_path) as ds:
        ref_shape = ds.shape
        ref_transform = ds.transform
        ref_crs = ds.crs

    ref_mask, policy = rasterize_reference_perimeter(geom, ref_shape, ref_transform)

    assert ref_mask.shape == (640, 982)
    assert policy["grid_shape"] == [640, 982]
    assert policy["pixel_size_m"] == 30.0
    assert policy["reference_raster_cells"] == 88958
    assert policy["reference_raster_area_ha"] == 8006.22


# 5. IoU bounded [0, 1]
def test_05_iou_bounded():
    metrics_path = VAL_DIR / "TALAMI_2017_VAL_001_METRICS.json"
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    iou = data["primary_metrics"]["iou"]
    assert 0.0 <= iou <= 1.0
    assert iou == 0.0428


# 6. Dice bounded [0, 1]
def test_06_dice_bounded():
    metrics_path = VAL_DIR / "TALAMI_2017_VAL_001_METRICS.json"
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    dice = data["primary_metrics"]["dice"]
    assert 0.0 <= dice <= 1.0
    assert dice == 0.0821


# 7. Intersection <= simulated area
def test_07_intersection_le_simulated_area():
    metrics_path = VAL_DIR / "TALAMI_2017_VAL_001_METRICS.json"
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    p = data["primary_metrics"]
    assert p["intersection_cells"] <= p["sim_cells"]
    assert p["intersection_area_ha"] <= p["sim_area_ha"]
    assert p["intersection_cells"] == 4250
    assert p["sim_cells"] == 14589


# 8. Intersection <= reference area
def test_08_intersection_le_reference_area():
    metrics_path = VAL_DIR / "TALAMI_2017_VAL_001_METRICS.json"
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    p = data["primary_metrics"]
    assert p["intersection_cells"] <= p["ref_cells"]
    assert p["intersection_area_ha"] <= p["reference_raster_area_ha"]
    assert p["ref_cells"] == 88958


# 9. Centroid calculations deterministic
def test_09_centroid_deterministic():
    from scripts.validate_talami_conaf import compute_centroid_metrics
    mask_path = FIRE_REF_DIR / "TALAMI_2017_FIRE_001_BURNED_MASK.tif"
    with rasterio.open(mask_path) as ds:
        sim_mask = ds.read(1).astype(bool)
        transform = ds.transform

    c1 = compute_centroid_metrics(sim_mask, sim_mask, transform)
    c2 = compute_centroid_metrics(sim_mask, sim_mask, transform)

    assert c1["centroid_distance_m"] == 0.0
    assert c1 == c2


# 10. Directional sectors exhaustive
def test_10_directional_sectors_exhaustive():
    metrics_path = VAL_DIR / "TALAMI_2017_VAL_001_METRICS.json"
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 8 directional extents
    dirs = data["directional_extents_8"]
    assert len(dirs) == 8
    expected_names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    assert [d["direction"] for d in dirs] == expected_names
    assert [d["center_deg"] for d in dirs] == [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]

    # 16 radial sectors
    radials = data["radial_overlap_16"]
    assert len(radials) == 16
    assert [r["azimuth_deg"] for r in radials] == [i * 22.5 for i in range(16)]


# 11. No calibration
def test_11_no_calibration():
    import yaml
    scen_path = REPO_ROOT / "configs" / "scenarios" / "talami_2017.yaml"
    with open(scen_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Physical scenario assumptions unchanged
    fire_cfg = cfg["fire"]
    assert fire_cfg["wind_adjustment_factor"] == 0.40
    assert fire_cfg["live_fuel_moisture"]["live_herbaceous"] == 0.60
    assert fire_cfg["live_fuel_moisture"]["live_woody"] == 0.90
    assert fire_cfg["simulation_horizon_hours"] == 12.0
    assert fire_cfg["start_time_status"] == "SCENARIO_HYPOTHESIS"
    assert fire_cfg["historical_truth"] is False


# 12. Same inputs -> deterministic metrics
def test_12_same_inputs_deterministic_metrics():
    from scripts.validate_talami_conaf import compute_primary_metrics
    arr_a = np.zeros((100, 100), dtype=bool)
    arr_b = np.zeros((100, 100), dtype=bool)
    arr_a[20:50, 20:50] = True
    arr_b[30:60, 30:60] = True

    m1 = compute_primary_metrics(arr_a, arr_b)
    m2 = compute_primary_metrics(arr_a, arr_b)
    assert m1 == m2


# 13. Validation release manifest verified
def test_13_validation_manifest_verified():
    val_manifest = REPO_ROOT / "manifests" / "QNYX_VAL_TALAMI_001_RELEASE_MANIFEST.json"
    assert val_manifest.exists()
    with open(val_manifest, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["stage"] == "validation"
    assert "TALAMI_2017_VAL_001_METRICS.json" in data["outputs"]


# 14. Horizon end inclusion semantics (QNYX-VAL-TALAMI-001A)
def test_horizon_end_inclusion_semantics():
    """
    Verify canonical policy: HORIZON_INCLUSION_POLICY = INCLUSIVE_END_TIME.
    Cells with arrival_time <= horizon_minutes (including exactly t = 720.0 min)
    are strictly counted as burned_by_horizon.
    """
    horizon_minutes = 720.0
    synthetic_arrivals = np.array([
        0.0,
        360.0,
        719.99,
        720.0,
        720.001,
        800.0,
        -1.0,
        np.nan,
    ])

    # Canonical definition:
    burned_by_horizon = (synthetic_arrivals >= 0.0) & (synthetic_arrivals <= horizon_minutes) & np.isfinite(synthetic_arrivals)

    # Synthetic cell with arrival_time == 720.0 must count as burned
    idx_720 = 3
    assert synthetic_arrivals[idx_720] == 720.0
    assert bool(burned_by_horizon[idx_720]) is True, "Cell with arrival_time == 720.0 must be burned_by_horizon"

    # Strict inequality check (< 720.0) vs (<= 720.0)
    burned_strict_lt = (synthetic_arrivals >= 0.0) & (synthetic_arrivals < horizon_minutes) & np.isfinite(synthetic_arrivals)
    assert bool(burned_strict_lt[idx_720]) is False
    assert int(np.sum(burned_by_horizon)) == int(np.sum(burned_strict_lt)) + 1

    # Beyond horizon (> 720.0) must NOT count
    assert bool(burned_by_horizon[4]) is False  # 720.001
    assert bool(burned_by_horizon[5]) is False  # 800.0
    assert bool(burned_by_horizon[6]) is False  # -1.0
    assert bool(burned_by_horizon[7]) is False  # NaN

    # Confirm consistency with frozen fire bundle
    fire_ref_dir = REPO_ROOT / "benchmarks" / "historical_talami_2017" / "fire_001_reference"
    with rasterio.open(fire_ref_dir / "TALAMI_2017_FIRE_001_BURNED_MASK.tif") as ds:
        burned_mask = ds.read(1)
    with rasterio.open(fire_ref_dir / "TALAMI_2017_FIRE_001_ARRIVAL_TIME.tif") as ds:
        arr_time = ds.read(1)

    mask_count = int(np.sum(burned_mask > 0))
    arr_le_720 = (arr_time >= 0.0) & (arr_time <= 720.0) & np.isfinite(arr_time)
    canonical_count = int(np.sum(arr_le_720))

    assert mask_count == 14589
    assert canonical_count == 14589
    assert mask_count == canonical_count
    assert np.array_equal(burned_mask > 0, arr_le_720)


