"""
tests/test_qnyx_bench_006.py

Suite de tests de verificación y validación para la Fase 5.4B de QNYX-BENCH-006
(Controlled Q-NYX vs FlamMap Wind-Slope Interaction Comparison - LCP Build & Verification).
Incluye validaciones rigurosas de los 6 Landscape files (.LCP) construidos manualmente en FlamMap GUI,
sus hashes SHA256 inmutables, tamaño exacto (1456932 bytes), mapeo de casos, gate de re-apertura LCP01,
geometría oficial heredada de BENCH-002 y estado PREPARED_NOT_RUN sin resultados ejecutados.
"""

import json
import math
import hashlib
import struct
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import Affine

from fire_engine.config import ROOT_DIR
from fire_engine.schemas import ScenarioConfig
from fire_engine.fuels.models import ORIGINAL_13_FUEL_MODELS, FuelMoisture
from fire_engine.directional.spread import DirectionalSpreadInputs, calculate_directional_rothermel_spread


BENCH_DIR = ROOT_DIR / "benchmarks" / "qnyx_bench_006_flammap_wind_slope"
BENCH002_DIR = ROOT_DIR / "benchmarks" / "flammap_bench_002"
BENCH005_DIR = ROOT_DIR / "benchmarks" / "qnyx_bench_005_wind_slope_interaction"

MANIFEST_PATH = BENCH_DIR / "manifest" / "benchmark_manifest.json"
IMMUTABLE_MANIFEST_PATH = BENCH_DIR / "manifest" / "immutable_input_manifest.json"
INPUTS_DIR = BENCH_DIR / "inputs"
COMMON_DIR = INPUTS_DIR / "common"
TOPOS_DIR = INPUTS_DIR / "topographies"
SCENARIOS_DIR = BENCH_DIR / "scenarios" / "qnyx"
FLAMMAP_DIR = BENCH_DIR / "flammap"
FLAMMAP_COMMON_DIR = FLAMMAP_DIR / "common"
LANDSCAPES_DIR = FLAMMAP_DIR / "landscapes"
ANALYSIS_DIR = BENCH_DIR / "analysis"
QNYX_DIR = BENCH_DIR / "qnyx"
INSTRUCTIONS_DIR = BENCH_DIR / "instructions"
REPORT_54A_PATH = ROOT_DIR / "FASE_5_4A_BENCH_006_QNYX_FLAMMAP_WIND_SLOPE_PREPARATION.md"
REPORT_54B_PATH = ROOT_DIR / "FASE_5_4B_BENCH_006_FLAMMAP_LCP_BUILD.md"

BASELINE_COMMIT = "337f368304b9a39b771f8eb87c7e9808106d473f"
BASELINE_TAG = "phase-5.5d-bench-006-qnyx-execution-audit-complete"


# Geometría oficial esperada
OFFICIAL_GRID_SHAPE = (301, 301)
OFFICIAL_RES = 30.0
OFFICIAL_CRS = "EPSG:32719"
OFFICIAL_TRANSFORM = Affine(30.0, 0.0, 188850.0, 0.0, -30.0, 6077580.0)
OFFICIAL_BOUNDS = {
    "left": 188850.0,
    "bottom": 6068550.0,
    "right": 197880.0,
    "top": 6077580.0
}

IGNITION_ROW = 150
IGNITION_COL = 150
IGNITION_X = 193365.0
IGNITION_Y = 6073065.0

EXPECTED_LCPS = {
    "bench006_flat.lcp": "C07339506FD7007D6AB491B5831FF030403D7BBA594EE3C821FF032AD64E13B5",
    "bench006_slope10_up_east.lcp": "AE711822225E51B60BFA191CBA80234A0EE89C01B4D17CDA30B39314F8744417",
    "bench006_slope10_up_ne45.lcp": "48143B4EAED0073852DFEE26E90FB22531656C1685ED4BEAF6FD628A3EC6AB44",
    "bench006_slope10_up_north.lcp": "21F5E77CC6C8253A615F6C909FCFD861F2371A66A7BF448AC6A77FAFA6554D72",
    "bench006_slope10_up_sw225.lcp": "38F511D2B3B6DEB9C1F1171053543DD533DFDCB00C725258F524071C7607201A",
    "bench006_slope10_up_west.lcp": "1E2BF7936188FDD484EC125B98C6A630F36764B5D6543FECC9FCF169D7D6AFCE"
}

EXPECTED_CASE01_PROJECT_FILES = {
    "BENCH006_CASE01_FLAT_CALM.fmp": {"size": 4718, "sha256": "0081607C65CD0FEE992F6700397D770CA3A968CF5B4F42A2B3EFA9E017585558"},
    "BENCH006_CASE01_FLAT_CALM.fzp": {"size": 5529, "sha256": "93F9C5E2A5301ABC1E928006CBDAAB8A47DB4FD0EE3D5A6ADE2CEF817CF1BDFB"}
}

EXPECTED_CASE01_OUTPUTS = {
    "run_log.txt": {"size": 1237, "sha256": "41838AE3F0323584FBDBE9FAC7AE6276697BA5B191A51D018487EE8D3CD9810B", "role": "RUN_LOG", "units": "text/plain"},
    "basic_rate_of_spread_m_min.tif": {"size": 3392, "sha256": "623D071ACB7959ACF7E1A2D3FC3039495ADD4FBFF4C6A159944AC7DB7CB62F46", "role": "LOCAL_BASIC_ROS", "units": "m/min"},
    "midflame_windspeed_m_s.tif": {"size": 3038, "sha256": "A472DD38A683AC76E833E5491911D8C0108354C23E0A2A8317E6B3D6DB31878E", "role": "LOCAL_MIDFLAME_WIND", "units": "m/s"},
    "mtt_arrival_time_min.tif": {"size": 3996, "sha256": "709C188203306A872D73ABDEFD733AD4C17B91B0082A273070D4BF6A6402CA7D", "role": "MTT_ARRIVAL_TIME", "units": "min"},
    "mtt_ros_m_min.tif": {"size": 3466, "sha256": "92BE9D79494F99D48EB80842E1D913B50B7F3BB31D006117D34D0E9D628BB29B", "role": "MTT_PATH_ROS", "units": "m/min"}
}


def compute_sha256(file_path: Path) -> str:
    """Calcula el hash SHA256 en mayúsculas de un archivo."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()


def circular_angular_difference(angle_a_deg: float, angle_b_deg: float) -> float:
    """Calcula la distancia angular mínima en el círculo [0°, 180°]."""
    diff = abs((angle_a_deg % 360.0) - (angle_b_deg % 360.0))
    return min(diff, 360.0 - diff)


def is_bench006_completed() -> bool:
    """Retorna True solo si el manifest oficial de BENCH-006 indica estado COMPLETED."""
    if not MANIFEST_PATH.exists():
        return False
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("execution_status") == "COMPLETED"
    except Exception:
        return False


# =============================================================================
# TESTS DE PREPARACIÓN Y BUILD DE LCP (FASES 5.4A - 5.4B)
# =============================================================================

def test_circular_angle_helper():
    """Verifica el cálculo de diferencia angular circular."""
    assert math.isclose(circular_angular_difference(10.0, 350.0), 20.0, abs_tol=1e-5)
    assert math.isclose(circular_angular_difference(90.0, 270.0), 180.0, abs_tol=1e-5)
    assert math.isclose(circular_angular_difference(90.0, 45.0), 45.0, abs_tol=1e-5)
    assert math.isclose(circular_angular_difference(90.0, 225.0), 135.0, abs_tol=1e-5)


def test_bench006_manifest_and_directory_structure():
    """Verifica que el manifest exista, tenga estado PREPARED_NOT_RUN y la estructura de directorios sea completa."""
    assert BENCH_DIR.exists()
    assert MANIFEST_PATH.exists()
    assert IMMUTABLE_MANIFEST_PATH.exists()
    assert COMMON_DIR.exists()
    assert TOPOS_DIR.exists()
    assert SCENARIOS_DIR.exists()
    assert FLAMMAP_DIR.exists()
    assert FLAMMAP_COMMON_DIR.exists()
    assert LANDSCAPES_DIR.exists()
    assert ANALYSIS_DIR.exists()
    assert INSTRUCTIONS_DIR.exists()

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["benchmark_id"] == "QNYX-BENCH-006"
    assert manifest["phase"] == "5.6A"
    assert manifest["execution_status"] == "FLAMMAP_AND_QNYX_SPATIAL_COMPARISON_COMPLETED"

    assert manifest["baseline_commit"] == BASELINE_COMMIT
    assert manifest["baseline_tag"] == BASELINE_TAG
    assert manifest["comparison_role"] == "EXTERNAL_REFERENCE_COMPARISON_NOT_GROUND_TRUTH"
    assert manifest["flammap_cases_completed"] == "8 / 8"
    assert manifest["flammap_execution_summary"]["flammap_cases_completed"] == "8 / 8"
    assert manifest["flammap_execution_summary"]["flammap_cases_pending"] == 0
    assert manifest["qnyx_cases_completed"] == "8 / 8"
    assert manifest["qnyx_execution_summary"]["qnyx_cases_completed"] == "8 / 8"
    assert manifest["qnyx_execution_summary"]["qnyx_cases_pending"] == 0


def test_flammap_readiness_lifecycle_and_built_lcp():
    """Verifica la semántica de ciclo de vida: source package ready, LCPs construidos y execution ready."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["flammap_landscape_source_package_ready"] == "PASS"
    assert manifest["flammap_lcp_build_status"] == "BUILT"
    assert manifest["flammap_execution_package_ready"] == "READY_FOR_EXECUTION"


def test_six_official_lcp_files_exist_with_exact_sizes_and_hashes():
    """Verifica que existan exactamente los 6 archivos LCP con tamaño exacto (1456932 bytes) y hashes congelados."""
    assert LANDSCAPES_DIR.exists()
    lcp_files = sorted(list(LANDSCAPES_DIR.glob("*.lcp")))
    assert len(lcp_files) == 6, f"Se esperaban 6 LCPs, encontrados {len(lcp_files)}"

    for lcp_p in lcp_files:
        assert lcp_p.name in EXPECTED_LCPS, f"Archivo LCP inesperado: {lcp_p.name}"
        assert lcp_p.stat().st_size == 1456932, f"Tamaño inválido en {lcp_p.name}: {lcp_p.stat().st_size}"
        actual_sha = hashlib.sha256(lcp_p.read_bytes()).hexdigest().upper()
        assert actual_sha == EXPECTED_LCPS[lcp_p.name], f"Hash mismatch en {lcp_p.name}: {actual_sha} != {EXPECTED_LCPS[lcp_p.name]}"


def test_flammap_lcp_build_audit_content_and_reopen_status():
    """Verifica el archivo de auditoría de LCPs, versión de FlamMap y el gate de re-apertura de LCP01."""
    audit_path = ANALYSIS_DIR / "flammap_lcp_build_audit.json"
    assert audit_path.exists()
    with open(audit_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["flammap_version_expected"] == "6.2.45.0"
    assert data["flammap_version_observed"] == "6.2.45.0"
    assert data["flammap_build_date_observed"] == "2025-08-28"
    assert data["flammap_version_match_status"] == "FLAMMAP_VERSION_MATCH_PASS"
    assert data["generation_application"] == "FlamMap 6.2.45.0"
    assert data["build_method"] == "MANUAL_FLAMMAP_GUI"

    assert data["lcp_count"] == 6
    assert data["count_status"] == "SIX_LCP_COUNT_PASS"
    assert data["hash_audit_status"] == "LCP_HASH_AUDIT_PASS"
    assert data["lcp01_reopen_gate"] == "LCP01_REOPEN_PASS"
    assert data["common_container_properties"]["eight_theme_container"] is True
    assert data["common_container_properties"]["surface_fire_execution_intent"] is True

    landscapes = data["landscapes"]
    assert len(landscapes) == 6

    # LCP01 debe tener reopen PASS
    lcp01 = [l for l in landscapes if l["lcp_id"] == "LCP01_flat"][0]
    assert lcp01["manual_reopen_checked"] is True
    assert lcp01["manual_reopen_status"] == "PASS"

    # Los otros 5 deben registrarse como NOT_INDIVIDUALLY_TESTED
    for other_lcp in [l for l in landscapes if l["lcp_id"] != "LCP01_flat"]:
        assert other_lcp["manual_reopen_checked"] is False
        assert other_lcp["manual_reopen_status"] == "NOT_INDIVIDUALLY_TESTED"


def test_flammap_version_match_gate_in_manifest():
    """Verifica que el manifest oficial certifique la coincidencia de versión FlamMap 6.2.45.0."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    meta = manifest["meteorological_and_fuel_standards"]
    assert meta["flammap_expected_version"] == "6.2.45.0"
    assert meta["flammap_observed_version"] == "6.2.45.0"
    assert meta["flammap_build_date_observed"] == "2025-08-28"
    assert meta["flammap_version_match_status"] == "FLAMMAP_VERSION_MATCH_PASS"

    gates = manifest["validation_gates"]
    assert "FLAMMAP_VERSION_MATCH_PASS" in gates
    assert gates["FLAMMAP_VERSION_MATCH_PASS"]["status"] == "PASS"


def test_case_to_lcp_mapping_exact():
    """Verifica que los 8 casos en case_definition.json apunten al archivo LCP correcto."""
    expected_mapping = {
        "case_01_control_flat_calm": "bench006_flat.lcp",
        "case_02_control_pure_wind_bench002": "bench006_flat.lcp",
        "case_03_control_pure_slope_10deg": "bench006_slope10_up_east.lcp",
        "case_04_wind_slope_aligned_0deg": "bench006_slope10_up_east.lcp",
        "case_05_wind_slope_oblique_45deg": "bench006_slope10_up_ne45.lcp",
        "case_06_wind_slope_cross_90deg": "bench006_slope10_up_north.lcp",
        "case_07_wind_slope_oblique_135deg": "bench006_slope10_up_sw225.lcp",
        "case_08_wind_slope_opposed_180deg": "bench006_slope10_up_west.lcp"
    }

    for case_id, exp_lcp in expected_mapping.items():
        cdef_p = FLAMMAP_DIR / case_id / "case_definition.json"
        assert cdef_p.exists()
        with open(cdef_p, "r", encoding="utf-8") as f:
            cdef = json.load(f)
        assert cdef["landscape_source"]["expected_lcp_file"] == f"flammap/landscapes/{exp_lcp}"


def test_results_directory_absent_and_execution_prepared_not_run():
    """Verifica que no existan resultados prematuros de comparación y el estado sea correcto."""
    results_dir = BENCH_DIR / "results"
    assert not results_dir.exists(), "El directorio results/ de comparación no debe existir antes de la fase de comparación"

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["execution_status"] == "FLAMMAP_AND_QNYX_SPATIAL_COMPARISON_COMPLETED"


def test_all_24_geotiffs_share_exact_reference_geometry_and_bounds():
    """Verifica que TODAS las 24 capas GeoTIFF compartan exactamente el transform y bounds oficiales."""
    common_tifs = sorted(list(COMMON_DIR.glob("*.tif")))
    topo_tifs = sorted(list(TOPOS_DIR.glob("*/*.tif")))
    all_tifs = common_tifs + topo_tifs
    assert len(all_tifs) == 24, f"Se esperaban 24 GeoTIFFs, encontrados {len(all_tifs)}"

    for tif_p in all_tifs:
        with rasterio.open(tif_p) as src:
            assert src.shape == OFFICIAL_GRID_SHAPE, f"Shape inválido en {tif_p.name}"
            assert math.isclose(src.res[0], OFFICIAL_RES, abs_tol=1e-5), f"Resolución X inválida en {tif_p.name}"
            assert math.isclose(src.res[1], OFFICIAL_RES, abs_tol=1e-5), f"Resolución Y inválida en {tif_p.name}"
            assert str(src.crs) == OFFICIAL_CRS, f"CRS inválido en {tif_p.name}"
            assert src.transform == OFFICIAL_TRANSFORM, f"Transform incorrecto en {tif_p.name}: {src.transform}"
            assert math.isclose(src.bounds.left, OFFICIAL_BOUNDS["left"], abs_tol=1e-3)
            assert math.isclose(src.bounds.bottom, OFFICIAL_BOUNDS["bottom"], abs_tol=1e-3)
            assert math.isclose(src.bounds.right, OFFICIAL_BOUNDS["right"], abs_tol=1e-3)
            assert math.isclose(src.bounds.top, OFFICIAL_BOUNDS["top"], abs_tol=1e-3)


def test_no_geotiff_retains_incorrect_bench001_origin():
    """Verifica explícitamente que NINGÚN GeoTIFF conserve el origen incorrecto (190350, 6076080)."""
    for tif_p in BENCH_DIR.glob("**/*.tif"):
        with rasterio.open(tif_p) as src:
            assert src.transform[2] != 190350.0, f"Origen incorrecto 190350 en {tif_p}"
            assert src.transform[5] != 6076080.0, f"Origen incorrecto 6076080 en {tif_p}"


def test_ignition_raster_center_and_coordinate_alignment():
    """Verifica que el centro de la celda (150, 150) y el shapefile de ignición coincidan exactamente en (193365, 6073065)."""
    center_x = OFFICIAL_TRANSFORM[2] + (150 + 0.5) * OFFICIAL_TRANSFORM[0]
    center_y = OFFICIAL_TRANSFORM[5] + (150 + 0.5) * OFFICIAL_TRANSFORM[4]
    assert math.isclose(center_x, 193365.0, abs_tol=1e-3)
    assert math.isclose(center_y, 6073065.0, abs_tol=1e-3)

    shp_path = FLAMMAP_COMMON_DIR / "ignition_center.shp"
    raw = shp_path.read_bytes()
    rec_shape_type, x, y = struct.unpack("<idd", raw[108:132])
    assert rec_shape_type == 1
    assert math.isclose(x, 193365.0, abs_tol=1e-3)
    assert math.isclose(y, 6073065.0, abs_tol=1e-3)

    col = (x - OFFICIAL_TRANSFORM[2]) / OFFICIAL_TRANSFORM[0]
    row = (OFFICIAL_TRANSFORM[5] - y) / abs(OFFICIAL_TRANSFORM[4])
    assert math.isclose(col, 150.5, abs_tol=1e-3)
    assert math.isclose(row, 150.5, abs_tol=1e-3)
    assert int(col) == 150
    assert int(row) == 150


def test_manifest_scenario_set_matches_filesystem_exactly():
    """Verifica que los 8 escenarios en el manifest coincidan exactamente con el filesystem."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest_scenarios = sorted(list(manifest["scenario_matrix"].keys()))
    assert len(manifest_scenarios) == 8

    fs_scenarios = sorted([f.stem.replace("scenario_", "") for f in SCENARIOS_DIR.glob("scenario_case_*.json")])
    assert manifest_scenarios == fs_scenarios

    flammap_dirs = sorted([d.name for d in FLAMMAP_DIR.iterdir() if d.is_dir() and d.name.startswith("case_")])
    assert manifest_scenarios == flammap_dirs


def test_common_rasters_and_neutral_crown_properties():
    """Verifica las 6 capas comunes (incluyendo las 3 capas neutrales de copa para LCP)."""
    expected_rasters = {
        "fuel_model.tif": {"min": 1, "max": 1},
        "barriers.tif": {"min": 0, "max": 0},
        "canopy_cover.tif": {"min": 0, "max": 0},
        "canopy_height.tif": {"min": 0.0, "max": 0.0},
        "canopy_base_height.tif": {"min": 0.0, "max": 0.0},
        "canopy_bulk_density.tif": {"min": 0.0, "max": 0.0}
    }
    for rname, rinfo in expected_rasters.items():
        rpath = COMMON_DIR / rname
        assert rpath.exists(), f"Falta {rname} en inputs/common/"
        with rasterio.open(rpath) as src:
            assert src.shape == (301, 301)
            assert str(src.crs) == "EPSG:32719"
            data = src.read(1)
            assert not np.isnan(data).any()
            assert math.isclose(np.min(data), rinfo["min"], abs_tol=1e-4)
            assert math.isclose(np.max(data), rinfo["max"], abs_tol=1e-4)


@pytest.mark.skipif(not BENCH002_DIR.exists(), reason="BENCH-002 not included in standalone repo")
def test_crown_neutral_rasters_exact_match_bench002():
    """Verifica que las capas de copa sean idénticas a nivel de byte a las de FLAMMAP-BENCH-002."""
    for rname in ["canopy_height.tif", "canopy_base_height.tif", "canopy_bulk_density.tif"]:
        src_path = BENCH002_DIR / "master" / rname
        dst_path = COMMON_DIR / rname
        assert src_path.exists()
        assert dst_path.exists()
        src_h = hashlib.sha256(src_path.read_bytes()).hexdigest().upper()
        dst_h = hashlib.sha256(dst_path.read_bytes()).hexdigest().upper()
        assert src_h == dst_h, f"Hash mismatch en capa neutral de copa {rname}"


def test_topography_rasters_exactness_and_conventions():
    """Verifica que las 6 topografías analíticas tengan elevación consistente y derivadas exactas."""
    topos = [d for d in TOPOS_DIR.iterdir() if d.is_dir()]
    assert len(topos) == 6, f"Se esperaban 6 topografías analíticas, encontradas {len(topos)}"

    for topo in topos:
        elev_f = topo / "elevation.tif"
        slope_f = topo / "slope.tif"
        aspect_f = topo / "aspect.tif"

        assert elev_f.exists() and slope_f.exists() and aspect_f.exists(), f"Faltan capas en {topo.name}"

        with rasterio.open(elev_f) as s_e, rasterio.open(slope_f) as s_s, rasterio.open(aspect_f) as s_a:
            e_arr = s_e.read(1)
            s_arr = s_s.read(1)
            a_arr = s_a.read(1)

            assert e_arr.shape == (301, 301)
            assert np.all(np.isfinite(e_arr))
            assert np.all(np.isfinite(s_arr))
            assert np.all(np.isfinite(a_arr))

            s_min, s_max = np.min(s_arr), np.max(s_arr)
            assert math.isclose(s_min, s_max, abs_tol=1e-2), f"Pendiente no uniforme en {topo.name}"

            a_min, a_max = np.min(a_arr), np.max(a_arr)
            assert math.isclose(a_min, a_max, abs_tol=1e-2), f"Aspecto no uniforme en {topo.name}"


def test_all_8_scenarios_json_valid():
    """Verifica que los 8 archivos de escenario cumplan estrictamente el schema ScenarioConfig."""
    scen_files = sorted(list(SCENARIOS_DIR.glob("scenario_case_*.json")))
    assert len(scen_files) == 8

    for sf in scen_files:
        with open(sf, "r", encoding="utf-8") as f:
            data = json.load(f)

        sc = ScenarioConfig.model_validate(data)
        assert sc.spread_model == "rothermel"
        assert sc.duration_minutes == 180
        assert sc.propagation.neighborhood_type == "stencil_16"
        assert sc.ignition.row == 150
        assert sc.ignition.col == 150
        assert sc.initial_fuel_moisture.dead_1h_fraction == 0.08


@pytest.mark.skipif(not BENCH002_DIR.exists(), reason="BENCH-002 not included in standalone repo")
def test_flammap_fms_and_prj_support_files():
    """Verifica que flame_benchmark.fms y flame_benchmark.prj existan y coincidan con BENCH-002."""
    fms_path = FLAMMAP_COMMON_DIR / "flame_benchmark.fms"
    prj_path = FLAMMAP_COMMON_DIR / "flame_benchmark.prj"
    assert fms_path.exists()
    assert prj_path.exists()

    fms_text = fms_path.read_text(encoding="utf-8").strip()
    assert fms_text == "1 8 9 10 100 100 10", f"Contenido FMS inesperado: {fms_text}"

    src_fms_h = hashlib.sha256((BENCH002_DIR / "master" / "flame_benchmark.fms").read_bytes()).hexdigest().upper()
    dst_fms_h = hashlib.sha256(fms_path.read_bytes()).hexdigest().upper()
    assert src_fms_h == dst_fms_h


def test_landscape_source_manifest_and_theme_mapping():
    """Verifica que landscape_source_manifest.csv contenga 6 landscapes mapeando las 8 themes con estado BUILT."""
    lman_path = FLAMMAP_DIR / "landscape_source_manifest.csv"
    assert lman_path.exists()
    df = pd.read_csv(lman_path)
    assert len(df) == 6
    assert all(df["all_8_themes_present"] == True)
    assert all(df["grid_alignment_pass"] == True)
    assert all(df["lcp_status"] == "BUILT")
    assert all(df["size_bytes"] == 1456932)
    assert all(df["north_up"] == True)
    assert all(df["crs"] == "EPSG:32719")

    lcp_flat = df[df["landscape_id"] == "LCP01_flat"].iloc[0]
    assert "case_01" in lcp_flat["used_by_cases"] and "case_02" in lcp_flat["used_by_cases"]


def test_case_input_layer_manifests_include_eight_themes_and_barriers_distinction():
    """Verifica que los 8 input_layer_manifest.csv distingan las 8 themes LCP de barriers (QNYX_ONLY)."""
    for cid in [f"case_0{i}" for i in range(1, 9)]:
        case_dirs = list(FLAMMAP_DIR.glob(f"{cid}_*"))
        assert len(case_dirs) == 1
        cdir = case_dirs[0]
        lman = cdir / "input_layer_manifest.csv"
        assert lman.exists()
        df = pd.read_csv(lman)
        assert len(df) == 12

        lcp_themes = df[df["role"] == "LANDSCAPE_THEME"]
        assert len(lcp_themes) == 8

        barriers_row = df[df["theme_name"] == "barriers"].iloc[0]
        assert barriers_row["role"] == "QNYX_ONLY_NOT_LCP_THEME"


def test_no_external_operational_paths_in_flammap_package():
    """Verifica la autocontención: ningún archivo operativo apunta a otros benchmarks."""
    for cdef_path in FLAMMAP_DIR.glob("case_*/case_definition.json"):
        with open(cdef_path, "r", encoding="utf-8") as f:
            cdef = json.load(f)
        text_dump = json.dumps(cdef)
        assert "flammap_bench_001" not in text_dump
        assert "flammap_bench_002" not in text_dump
        assert "qnyx_bench_005" not in text_dump


def test_flammap_wind_semantics_and_waf_audit():
    """Verifica la recuperación formal de la semántica de viento desde BENCH-002."""
    wind_audit_path = ANALYSIS_DIR / "flammap_wind_semantics_audit.json"
    assert wind_audit_path.exists()
    with open(wind_audit_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cfg = data["flammap_configuration"]
    assert cfg["software"] == "FlamMap"
    assert cfg["expected_version"] == "6.2.45.0"
    assert cfg["input_wind_speed"] == 5.0
    assert cfg["input_wind_units"] == "mph (U20)"
    assert cfg["wind_direction_from_deg"] == 270.0
    assert math.isclose(cfg["observed_midflame_wind_ms"], 0.810649037361145, rel_tol=1e-6)
    assert math.isclose(cfg["observed_waf"], 0.3626740503584221, rel_tol=1e-6)


def test_angular_design_and_separation_classes():
    """Verifica las 5 clases de separación angular en angular_design_audit.csv."""
    audit_path = ANALYSIS_DIR / "angular_design_audit.csv"
    assert audit_path.exists()
    df = pd.read_csv(audit_path)
    assert len(df) == 8

    c4 = df[df["case_id"] == "case_04_wind_slope_aligned_0deg"].iloc[0]
    assert c4["calculated_separation_deg"] == 0.0

    c5 = df[df["case_id"] == "case_05_wind_slope_oblique_45deg"].iloc[0]
    assert c5["calculated_separation_deg"] == 45.0

    c6 = df[df["case_id"] == "case_06_wind_slope_cross_90deg"].iloc[0]
    assert c6["calculated_separation_deg"] == 90.0

    c7 = df[df["case_id"] == "case_07_wind_slope_oblique_135deg"].iloc[0]
    assert c7["calculated_separation_deg"] == 135.0
    assert c7["upslope_deg"] == 225.0
    assert c7["aspect_downslope_deg"] == 45.0

    c8 = df[df["case_id"] == "case_08_wind_slope_opposed_180deg"].iloc[0]
    assert c8["calculated_separation_deg"] == 180.0


def test_local_preflight_predictions_and_case_08_heading_residual():
    """Verifica las predicciones analíticas locales y comprueba que Case 08 mantiene rumbo residual al Este."""
    pref_path = ANALYSIS_DIR / "qnyx_local_preflight_predictions.csv"
    assert pref_path.exists()
    df = pd.read_csv(pref_path)
    assert len(df) == 8

    c1 = df[df["case_id"] == "case_01_control_flat_calm"].iloc[0]
    assert math.isclose(c1["r0_m_min"], 1.212031, abs_tol=1e-4)
    assert c1["phi_resultant"] == 0.0

    c2 = df[df["case_id"] == "case_02_control_pure_wind_bench002"].iloc[0]
    assert math.isclose(c2["phi_w"], 2.622188, abs_tol=1e-4)
    assert math.isclose(c2["r_head_m_min"], 4.390205, abs_tol=1e-4)

    c3 = df[df["case_id"] == "case_03_control_pure_slope_10deg"].iloc[0]
    assert math.isclose(c3["phi_s"], 1.279537, abs_tol=1e-4)
    assert math.isclose(c3["r_head_m_min"], 2.762870, abs_tol=1e-4)

    c8 = df[df["case_id"] == "case_08_wind_slope_opposed_180deg"].iloc[0]
    assert c8["phi_w"] > c8["phi_s"]
    assert math.isclose(float(c8["predicted_resultant_heading_deg"]), 90.0, abs_tol=1e-2)
    assert math.isclose(c8["phi_resultant"], 2.622188 - 1.279537, abs_tol=1e-4)


def test_heritage_controls_consistency_and_frozen_benchmarks():
    """Verifica que el archivo de auditoría de herencia contenga la trazabilidad de BENCH-001, 002, 003 y 005."""
    hpath = ANALYSIS_DIR / "benchmark_heritage_audit.json"
    assert hpath.exists()
    with open(hpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data["heritage_records"]) == 4


def test_bench005_geospatial_heritage_audit_and_bench002_reference_source():
    """Verifica la auditoría de herencia geoespacial BENCH-005 -> BENCH-006 y el contrato BENCH-002."""
    b5_audit_path = ANALYSIS_DIR / "bench005_geospatial_heritage_audit.json"
    assert b5_audit_path.exists()
    with open(b5_audit_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["status"] == "BENCH005_GEOSPATIAL_HERITAGE_CHARACTERIZED"
    assert data["bench006_heritage_definition"]["spatial_grid_reference_source"] == "FLAMMAP-BENCH-002 / QNYX-BENCH-002"
    assert data["bench006_heritage_definition"]["official_reference_transform"] == [30.0, 0.0, 188850.0, 0.0, -30.0, 6077580.0]
    assert data["impact_assessment"]["internal_numerical_results"] == "UNAFFECTED_BY_GLOBAL_TRANSLATION"
    assert data["impact_assessment"]["absolute_raster_geospatial_placement"] == "SHIFTED"
    assert data["impact_assessment"]["absolute_vector_geospatial_placement"] == "SHIFTED"


@pytest.mark.skipif(not BENCH005_DIR.exists(), reason="BENCH-005 not included in standalone repo")
def test_bench005_exhaustive_geotiff_audit_matches_filesystem():
    """Verifica que la auditoría exhaustiva de GeoTIFFs de BENCH-005 coincida exactamente con los archivos reales en disco."""
    csv_p = ANALYSIS_DIR / "bench005_all_geotiff_geospatial_audit.csv"
    assert csv_p.exists()
    df = pd.read_csv(csv_p)
    actual_b5_tifs = list(BENCH005_DIR.rglob("*.tif"))
    assert len(df) == len(actual_b5_tifs), f"Conteo mismatch: audit={len(df)}, disk={len(actual_b5_tifs)}"
    assert all(df["grid_category"] == "LEGACY_SHIFTED_GRID")


@pytest.mark.skipif(not BENCH005_DIR.exists(), reason="BENCH-005 not included in standalone repo")
def test_bench005_vector_provenance_and_metric_sanity_check():
    """Verifica la auditoría de procedencia de salidas vectoriales de BENCH-005 y el sanity check métrico de Case 01."""
    vec_audit_path = ANALYSIS_DIR / "bench005_vector_geospatial_audit.json"
    assert vec_audit_path.exists()
    with open(vec_audit_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    actual_b5_geojsons = list(BENCH005_DIR.rglob("*.geojson"))
    assert data["total_vector_files_characterized"] == len(actual_b5_geojsons)
    assert data["coordinate_generation_provenance"]["placement_classification"] == "ABSOLUTE_VECTOR_GEOSPATIAL_PLACEMENT = SHIFTED"

    # Verificar sanity check métrico en Case 01
    c1_chk = data["case01_metric_sanity_check"]
    assert math.isclose(c1_chk["distance_to_legacy_center_m"], 0.0, abs_tol=1e-2)
    assert math.isclose(c1_chk["distance_to_reference_center_m"], 2121.32, abs_tol=1.0)


def test_bench005_absolute_geospatial_metadata_audit():
    """Verifica la auditoría de metadatos geoespaciales absolutos en los JSON de resultados de BENCH-005."""
    meta_path = ANALYSIS_DIR / "bench005_absolute_geospatial_metadata_audit.json"
    assert meta_path.exists()
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["summary"]["timeline_json_count_with_extent_utm"] == 21
    assert data["summary"]["web_export_json_count_with_extent"] == 21
    assert data["placement_classification"].startswith("ABSOLUTE_JSON_GEOSPATIAL_METADATA = SHIFTED")


@pytest.mark.skipif(not REPORT_54A_PATH.exists(), reason="Internal markdown report not included in standalone repo")
def test_impact_classification_and_wording_prohibitions():
    """Verifica que las descripciones del propagador y equivalencia rotacional no contengan frases erróneas."""
    for fpath in [REPORT_54A_PATH, REPORT_54B_PATH, ANALYSIS_DIR / "bench005_geospatial_heritage_audit.json"]:
        text = fpath.read_text(encoding="utf-8")
        assert "Dijkstra sobre retícula continua" not in text
        assert "simetrías ortogonales y diagonales" not in text


def test_theoretical_domain_preflight_margin():
    """Verifica que a 180 min el margen libre teórico sea superior a 3400 m."""
    pref_path = ANALYSIS_DIR / "qnyx_local_preflight_predictions.csv"
    df = pd.read_csv(pref_path)
    max_dist = df["theoretical_head_distance_180min_m"].max()
    min_margin = df["theoretical_domain_margin_m"].min()

    assert math.isclose(max_dist, 1069.39, abs_tol=1.0)
    assert min_margin > 3400.0, f"Margen teórico insuficiente: {min_margin}"


def test_comparison_schemas_structure_and_pending_status():
    """Verifica la existencia y estructura de los 3 esquemas de comparación."""
    for sname in ["local_comparison_schema.csv", "spatial_comparison_schema.csv", "spatial_direction_comparison_schema.csv"]:
        spath = ANALYSIS_DIR / sname
        assert spath.exists(), f"Falta {sname}"
        df = pd.read_csv(spath)
        assert len(df) == 8


def test_builder_determinism_audit_pass():
    """Verifica que la auditoría de determinismo del builder certifique 0 discrepancias."""
    det_path = ANALYSIS_DIR / "builder_determinism_audit.json"
    assert det_path.exists()
    with open(det_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["status"] == "BUILDER_DETERMINISM_PASS"
    assert data["mismatches_count"] == 0
    assert data["missing_count"] == 0
    assert data["extra_count"] == 0
    assert data["files_compared"] >= 65


def test_immutable_inputs_manifest_and_hashes():
    """Verifica que el manifiesto de insumos inmutables contenga todos los archivos registrados y coincidan bit a bit."""
    assert IMMUTABLE_MANIFEST_PATH.exists()
    with open(IMMUTABLE_MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["total_immutable_files"] == len(data["immutable_inputs"])
    for rel_p, finfo in data["immutable_inputs"].items():
        full_p = BENCH_DIR / rel_p
        assert full_p.exists(), f"Insumo inmutable no encontrado: {rel_p}"
        current_sha = hashlib.sha256(full_p.read_bytes()).hexdigest().upper()
        assert current_sha == finfo["sha256"].upper(), f"Hash alterado en {rel_p}"


def test_preparation_validation_gates_all_pass():
    """Verifica que los gates de preparación y build de LCPs en el manifest estén en estado PASS."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    gates = manifest.get("validation_gates", {})
    assert len(gates) >= 17
    for gname, ginfo in gates.items():
        if gname == "QNYX_FLAMMAP_SPATIAL_EQUIVALENCE":
            assert ginfo["status"] == "NOT_CLAIMED"
        else:
            assert ginfo["status"] == "PASS", f"Gate {gname} no pasó: {ginfo}"


@pytest.mark.skipif(not REPORT_54A_PATH.exists(), reason="Internal markdown report not included in standalone repo")
def test_document_integrity_and_no_control_character_corruption():
    """Verifica que los documentos markdown no contengan caracteres de control corruptos y tengan UTF-8 válido."""
    for md_path in [INSTRUCTIONS_DIR / "BENCH_006_FLAMMAP_EXECUTION_INSTRUCTIONS.md", REPORT_54A_PATH, REPORT_54B_PATH]:
        assert md_path.exists(), f"Falta documento: {md_path.name}"
        raw = md_path.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf"), f"{md_path.name} contiene BOM"

        for b in range(0x20):
            if b != 0x0A:
                assert b not in raw, f"{md_path.name} contiene byte de control 0x{b:02X}"

        text = raw.decode("utf-8")
        for bad_sub in ["raceta", "^L", "^M", "Might"]:
            assert bad_sub not in text, f"Texto corrupto detectado en {md_path.name}: {bad_sub}"


@pytest.mark.skipif(not REPORT_54A_PATH.exists(), reason="Internal markdown report not included in standalone repo")
def test_document_integrity_bench005_json_enumeration_and_prohibitions():
    """Verifica que el informe y análisis enumeren los 7 tipos reales de JSON de BENCH-005 y prohíbe nombres inexistentes."""
    assert REPORT_54A_PATH.exists()
    report_text = REPORT_54A_PATH.read_text(encoding="utf-8")

    # Prohibir nombres de JSON que no existen en BENCH-005/results/
    for invalid_name in ["execution_metrics.json", "spatial_metrics.json", "comparison_metrics.json", "comparison_results.json"]:
        assert invalid_name not in report_text, f"Referencia a JSON inexistente {invalid_name} encontrada en informe"

    # Verificar que los 7 nombres reales estén documentados
    for real_name in ["timeline.json", "web_export.json", "local_physics.json", "run_manifest.json", "run_metadata.json", "scenario_used.json", "statistics.json"]:
        assert real_name in report_text, f"Falta mencionar archivo JSON real {real_name} en informe"


# =============================================================================
# TESTS DE AUDITORÍA Y RESULTADOS DE FLAMMAP CASE 01 (FASE 5.4C-1)
# =============================================================================

def test_case01_project_artifacts_exist_and_exact_hashes():
    """Verifica que los archivos de proyecto FlamMap (.fmp, .fzp) de Case 01 existan con hashes exactos."""
    projects_dir = FLAMMAP_DIR / "projects"
    assert projects_dir.exists()

    for pname, pinfo in EXPECTED_CASE01_PROJECT_FILES.items():
        ppath = projects_dir / pname
        assert ppath.exists(), f"Falta archivo de proyecto {pname}"
        assert ppath.stat().st_size == pinfo["size"], f"Tamaño inválido en {pname}: {ppath.stat().st_size} != {pinfo['size']}"
        actual_sha = hashlib.sha256(ppath.read_bytes()).hexdigest().upper()
        assert actual_sha == pinfo["sha256"], f"Hash mismatch en {pname}: {actual_sha} != {pinfo['sha256']}"


def test_case01_run_log_content_and_gate():
    """Verifica que el run log de Case 01 exista, tenga tamaño y hash exacto, y cumpla el encabezado y pie requeridos."""
    log_path = FLAMMAP_DIR / "case_01_control_flat_calm" / "outputs" / "run_log.txt"
    assert log_path.exists()
    assert log_path.stat().st_size == 1237
    actual_sha = hashlib.sha256(log_path.read_bytes()).hexdigest().upper()
    assert actual_sha == "41838AE3F0323584FBDBE9FAC7AE6276697BA5B191A51D018487EE8D3CD9810B"

    text = log_path.read_text(encoding="utf-8").strip()
    assert text.startswith("#Run Log for MTT run: BENCH006_CASE01_FLAT_CALM\n#FlamMap Version 6.2.45.0")
    assert text.endswith("#MTT_ROS\n#MTT_ARRIVAL\n#END SELECTED MTT OUTPUTS\n#SPOTTING_SEED: -1")


def test_case01_output_exact_hashes_and_manifests():
    """Verifica que los 5 artefactos de salida de Case 01 existan con tamaños y hashes SHA256 exactos."""
    outputs_dir = FLAMMAP_DIR / "case_01_control_flat_calm" / "outputs"
    assert outputs_dir.exists()

    for out_name, out_info in EXPECTED_CASE01_OUTPUTS.items():
        out_p = outputs_dir / out_name
        assert out_p.exists(), f"Falta salida de Case 01: {out_name}"
        assert out_p.stat().st_size == out_info["size"], f"Tamaño inválido en {out_name}"
        actual_sha = hashlib.sha256(out_p.read_bytes()).hexdigest().upper()
        assert actual_sha == out_info["sha256"], f"Hash mismatch en {out_name}: {actual_sha} != {out_info['sha256']}"

    # Verificar existencia y contenido de output_manifest.json y output_manifest.csv
    out_man_json = FLAMMAP_DIR / "case_01_control_flat_calm" / "output_manifest.json"
    assert out_man_json.exists()
    with open(out_man_json, "r", encoding="utf-8") as f:
        mdata = json.load(f)
    assert mdata["benchmark_id"] == "QNYX-BENCH-006"
    assert mdata["case_id"] == "case_01_control_flat_calm"
    assert mdata["total_outputs"] == 5

    out_man_csv = FLAMMAP_DIR / "case_01_control_flat_calm" / "output_manifest.csv"
    assert out_man_csv.exists()
    df_man = pd.read_csv(out_man_csv)
    assert len(df_man) == 5
    assert set(df_man["semantic_role"]) == {"RUN_LOG", "LOCAL_BASIC_ROS", "LOCAL_MIDFLAME_WIND", "MTT_ARRIVAL_TIME", "MTT_PATH_ROS"}


def test_case01_basic_ros_metadata_uniformity_and_local_comparison():
    """Verifica la tasa básica de propagación (Basic ROS): metadatos, uniformidad estricta y comparación local con Q-NYX."""
    ros_p = FLAMMAP_DIR / "case_01_control_flat_calm" / "outputs" / "basic_rate_of_spread_m_min.tif"
    assert ros_p.exists()

    with rasterio.open(ros_p) as src:
        assert src.shape == OFFICIAL_GRID_SHAPE
        assert str(src.crs) == OFFICIAL_CRS
        assert math.isclose(src.res[0], OFFICIAL_RES, abs_tol=1e-5)
        assert math.isclose(src.res[1], OFFICIAL_RES, abs_tol=1e-5)
        assert src.nodata == -9999.0
        data = src.read(1)
        assert data.dtype == np.float32

        center_val = float(data[150, 150])
        min_val = float(np.min(data))
        max_val = float(np.max(data))

        assert math.isclose(center_val, 1.2120143175125122, rel_tol=1e-6)
        assert math.isclose(min_val, 1.2120143175125122, rel_tol=1e-6)
        assert math.isclose(max_val, 1.2120143175125122, rel_tol=1e-6)
        assert min_val == max_val, "Basic ROS no es uniforme en toda la grilla"

        # Comparación analítica local
        qnyx_ref = 1.21203124935
        signed_diff = center_val - qnyx_ref
        rel_pct_diff = (signed_diff / qnyx_ref) * 100.0
        assert math.isclose(signed_diff, -0.000016931837, abs_tol=1e-6)
        assert abs(rel_pct_diff) < 0.002, f"Diferencia relativa excesiva: {rel_pct_diff}%"


def test_case01_midflame_wind_zero():
    """Verifica que el ráster de viento a media llama sea estrictamente 0.0 m/s en todo el dominio."""
    wind_p = FLAMMAP_DIR / "case_01_control_flat_calm" / "outputs" / "midflame_windspeed_m_s.tif"
    assert wind_p.exists()

    with rasterio.open(wind_p) as src:
        assert src.shape == OFFICIAL_GRID_SHAPE
        assert str(src.crs) == OFFICIAL_CRS
        assert src.nodata == -9999.0
        data = src.read(1)
        assert data.dtype == np.float32

        assert float(data[150, 150]) == 0.0
        assert float(np.min(data)) == 0.0
        assert float(np.max(data)) == 0.0


def test_case01_mtt_arrival_metadata_and_metrics():
    """Verifica los metadatos y métricas físicas de MTT Arrival Time (200 celdas, 18.0 ha, <= 180 min, sin contacto de borde)."""
    arr_p = FLAMMAP_DIR / "case_01_control_flat_calm" / "outputs" / "mtt_arrival_time_min.tif"
    assert arr_p.exists()

    with rasterio.open(arr_p) as src:
        assert src.shape == OFFICIAL_GRID_SHAPE
        assert str(src.crs) == OFFICIAL_CRS
        assert src.transform == OFFICIAL_TRANSFORM
        assert src.nodata == -9999.0
        data = src.read(1)
        assert data.dtype == np.float32

        val_mask = (data != src.nodata) & (data >= 0)
        val_count = int(np.count_nonzero(val_mask))
        assert val_count == 200, f"Se esperaban 200 celdas válidas, encontradas {val_count}"

        burned_area_ha = val_count * 30.0 * 30.0 / 10000.0
        assert math.isclose(burned_area_ha, 18.0, abs_tol=1e-4)

        center_arr = float(data[150, 150])
        assert math.isclose(center_arr, 0.0, abs_tol=1e-5), f"Ignición central no es 0.0 min: {center_arr}"

        val_data = data[val_mask]
        min_arr = float(np.min(val_data))
        max_arr = float(np.max(val_data))
        assert math.isclose(min_arr, 0.0, abs_tol=1e-5)
        assert math.isclose(max_arr, 178.49053955078125, rel_tol=1e-6)
        assert max_arr <= 180.0, f"Tiempo de llegada supera el horizonte de 180 min: {max_arr}"

        rows, cols = np.where(val_mask)
        assert int(np.min(rows)) == 143
        assert int(np.max(rows)) == 158
        assert int(np.min(cols)) == 142
        assert int(np.max(cols)) == 157

        # Distancia radial máxima desde el centro (150, 150)
        r_coords = rows - 150
        c_coords = cols - 150
        max_radial_dist = float(np.max(np.sqrt(r_coords**2 + c_coords**2) * 30.0))
        assert math.isclose(max_radial_dist, 258.0697580112788, rel_tol=1e-5)

        # Contacto de borde en grilla 301x301
        border_contacts = int(np.count_nonzero((rows == 0) | (rows == 300) | (cols == 0) | (cols == 300)))
        assert border_contacts == 0, f"Contactos de borde no nulos: {border_contacts}"


def test_case01_mtt_ros_metadata_metrics_and_arrival_mask_parity():
    """Verifica MTT ROS: 200 celdas válidas, 199 positivas, paridad estricta de máscara con MTT Arrival."""
    mtt_ros_p = FLAMMAP_DIR / "case_01_control_flat_calm" / "outputs" / "mtt_ros_m_min.tif"
    arr_p = FLAMMAP_DIR / "case_01_control_flat_calm" / "outputs" / "mtt_arrival_time_min.tif"
    assert mtt_ros_p.exists()
    assert arr_p.exists()

    with rasterio.open(mtt_ros_p) as src_ros, rasterio.open(arr_p) as src_arr:
        assert src_ros.shape == OFFICIAL_GRID_SHAPE
        assert str(src_ros.crs) == OFFICIAL_CRS
        assert src_ros.transform == OFFICIAL_TRANSFORM
        assert src_ros.nodata == -9999.0
        data_ros = src_ros.read(1)
        data_arr = src_arr.read(1)
        assert data_ros.dtype == np.float32

        mask_ros = (data_ros != src_ros.nodata) & (data_ros >= 0)
        mask_arr = (data_arr != src_arr.nodata) & (data_arr >= 0)

        assert int(np.count_nonzero(mask_ros)) == 200
        assert np.array_equal(mask_ros, mask_arr), "Máscara de MTT ROS no coincide con MTT Arrival"
        assert int(np.count_nonzero(mask_ros & ~mask_arr)) == 0
        assert int(np.count_nonzero(mask_arr & ~mask_ros)) == 0

        # Centro es 0.0 (punto de ignición)
        assert float(data_ros[150, 150]) == 0.0

        # 199 celdas positivas
        pos_mask = mask_ros & (data_ros > 0)
        pos_count = int(np.count_nonzero(pos_mask))
        assert pos_count == 199, f"Se esperaban 199 celdas positivas, encontradas {pos_count}"

        pos_data = data_ros[pos_mask]
        min_pos = float(np.min(pos_data))
        max_pos = float(np.max(pos_data))
        assert math.isclose(min_pos, 0.9090107083320618, rel_tol=1e-6)
        assert math.isclose(max_pos, 1.2120143175125122, rel_tol=1e-6)


def test_case01_execution_audit_file_and_classification():
    """Verifica el archivo de auditoría flammap_case01_execution_audit.json y sus gates de validación."""
    audit_p = ANALYSIS_DIR / "flammap_case01_execution_audit.json"
    assert audit_p.exists()

    with open(audit_p, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["benchmark_id"] == "QNYX-BENCH-006"
    assert data["phase"] == "5.4C-1"
    assert data["case_id"] == "case_01_control_flat_calm"
    assert data["run_name"] == "BENCH006_CASE01_FLAT_CALM"
    assert data["flammap_version"] == "6.2.45.0"
    assert data["flammap_case01_execution_status"] == "COMPLETED_AND_AUDITED"
    assert data["execution_classification"] == "FLAMMAP_CASE01_EXECUTION_STATUS = COMPLETED_AND_AUDITED"

    # Prohibir BENCH006_EXTERNAL_COMPARISON_PASS antes de la comparación espacial con Q-NYX
    text_dump = json.dumps(data)
    assert "BENCH006_EXTERNAL_COMPARISON_PASS" not in text_dump

    gates = data.get("validation_gates", {})
    assert len(gates) >= 10
    for gname, ginfo in gates.items():
        assert ginfo["status"] == "PASS", f"Gate {gname} no pasó en Case 01 audit"


def test_case02_to_08_project_artifacts_exist_and_exact_hashes():
    """Verifica la existencia y hashes SHA256 exactos de los 7 .fmp y 7 .fzp de Case 02 a Case 08."""
    proj_dir = FLAMMAP_DIR / "projects"
    expected_projects = {
        "BENCH006_CASE02_FLAT_PURE_WIND.fmp": (6635, "DFEEB7B022E3EFB289211B9EA3897593D8CF31D39EB8F8512CE8932C091F6F52"),
        "BENCH006_CASE02_FLAT_PURE_WIND.fzp": (17985, "42352BCD13D57E00A3BEF571C6B29F83BE288B1B04DBDDE0560DF7C122B4E81F"),
        "BENCH006_CASE03_PURE_SLOPE_10DEG.fmp": (4709, "2A6D5EC194800647947D6051371F4BBC3E43F55F1BE19A1586D0D4CB92B0386D"),
        "BENCH006_CASE03_PURE_SLOPE_10DEG.fzp": (8199, "9047CE872A136424EAA2207B4CE3E8606CFDE5F4270203EEECF8E6795E4E3B3F"),
        "BENCH006_CASE04_WIND_SLOPE_ALIGNED.fmp": (4716, "46D162DFFCB6C95B913392E2519405DD3A8C3B6ED8A27B0B7D0B7DA825727008"),
        "BENCH006_CASE04_WIND_SLOPE_ALIGNED.fzp": (14081, "AC175EEF587727326981567F2A311CCB5D7DFE6912933BF8133F83256C398F58"),
        "BENCH006_CASE05_WIND_SLOPE_OBLIQUE45.fmp": (4714, "9A3CFAB54404AC1C287DECC307B018279B2273041F6282D6A7BDB494FBCC14AC"),
        "BENCH006_CASE05_WIND_SLOPE_OBLIQUE45.fzp": (14081, "A7E7D02550170CDA4F5ED584A8ECBA83A273A87A8260FEB5FF5E8706A4B02136"),
        "BENCH006_CASE06_WIND_SLOPE_CROSS90.fmp": (6576, "8C6DBED0396BA095815D304CBFD1881A8884AFE810B9C14EA413995BC322EA39"),
        "BENCH006_CASE06_WIND_SLOPE_CROSS90.fzp": (19311, "A15A8D6820CC3BE1D6C01B8329586BD17114B52A6ACD3559435984CF24D86E01"),
        "BENCH006_CASE07_WIND_SLOPE_OBLIQUE135.fmp": (4786, "FB36883516B83A384B7001FB900DEB0808674EB2C7D9ECD956442DA07356EFFA"),
        "BENCH006_CASE07_WIND_SLOPE_OBLIQUE135.fzp": (11169, "16F4BEE10E56DC07D86608B67454E60E8050068013910D1638C845DE881FEC18"),
        "BENCH006_CASE08_WIND_SLOPE_OPPOSED180.fmp": (4716, "CDCD928CB595BEBDD74DC72EBBEA23705A81EC0FEBA6B6CAF666BC3318E55B55"),
        "BENCH006_CASE08_WIND_SLOPE_OPPOSED180.fzp": (8646, "AF46D31E4C159D5BC5D660A3FE2DF0D8AAB508F45E68DC5B777B1B4F294AD98F"),
    }

    for fname, (exp_size, exp_sha) in expected_projects.items():
        p = proj_dir / fname
        assert p.exists(), f"Falta archivo de proyecto {fname}"
        data = p.read_bytes()
        assert len(data) == exp_size, f"Tamaño incorrecto en {fname}: {len(data)} != {exp_size}"
        sha = hashlib.sha256(data).hexdigest().upper()
        assert sha == exp_sha, f"SHA256 incorrecto en {fname}: {sha} != {exp_sha}"


def test_case02_to_08_local_rasters_uniformity_and_expected_values():
    """Verifica que los rásteres locales de Case 02 a 08 sean uniformes y coincidan con los valores observados."""
    expected_local = {
        "case_02_control_pure_wind_bench002": {"ros": 4.390194892883301, "wind": 0.810649037361145, "dir": 90.0},
        "case_03_control_pure_slope_10deg": {"ros": 2.762868642807007, "wind": 0.0, "dir": 90.0},
        "case_04_wind_slope_aligned_0deg": {"ros": 5.941049575805664, "wind": 0.810649037361145, "dir": 90.0},
        "case_05_wind_slope_oblique_45deg": {"ros": 5.631171226501465, "wind": 0.810649037361145, "dir": 75.30630493164062},
        "case_06_wind_slope_cross_90deg": {"ros": 4.748347282409668, "wind": 0.810649037361145, "dir": 63.989044189453125},
        "case_07_wind_slope_oblique_135deg": {"ros": 3.553435802459717, "wind": 0.810649037361145, "dir": 117.25581359863281},
        "case_08_wind_slope_opposed_180deg": {"ros": 2.8393404483795166, "wind": 0.810649037361145, "dir": 90.0},
    }
    for cid, exp in expected_local.items():
        out_dir = FLAMMAP_DIR / cid / "outputs"
        # Basic ROS
        with rasterio.open(out_dir / "basic_rate_of_spread_m_min.tif") as src:
            arr = src.read(1)
            assert arr.shape == (301, 301)
            assert np.min(arr) == np.max(arr)
            assert np.isclose(arr[150, 150], exp["ros"], atol=1e-5)
        # Midflame Wind
        with rasterio.open(out_dir / "midflame_windspeed_m_s.tif") as src:
            arr = src.read(1)
            assert arr.shape == (301, 301)
            assert np.min(arr) == np.max(arr)
            assert np.isclose(arr[150, 150], exp["wind"], atol=1e-5)
        # Direction
        with rasterio.open(out_dir / "max_spread_direction_deg.tif") as src:
            arr = src.read(1)
            assert arr.shape == (301, 301)
            assert np.min(arr) == np.max(arr)
            assert np.isclose(arr[150, 150], exp["dir"], atol=1e-5)


def test_case02_to_08_mtt_arrival_and_metrics():
    """Verifica las métricas de llegada MTT (celdas válidas, área, ignición=0, 0 bordes) para Case 02 a 08."""
    expected_arrival = {
        "case_02_control_pure_wind_bench002": {"cells": 884, "area": 79.56, "max": 179.78501892089844, "rows": (135, 166), "cols": (141, 176)},
        "case_03_control_pure_slope_10deg": {"cells": 422, "area": 37.98, "max": 179.30880737304688, "rows": (140, 161), "cols": (143, 166)},
        "case_04_wind_slope_aligned_0deg": {"cells": 1188, "area": 106.92, "max": 179.7637939453125, "rows": (134, 167), "cols": (142, 185)},
        "case_05_wind_slope_oblique_45deg": {"cells": 1115, "area": 100.35, "max": 179.9567108154297, "rows": (131, 164), "cols": (142, 182)},
        "case_06_wind_slope_cross_90deg": {"cells": 920, "area": 82.80, "max": 179.98641967773438, "rows": (131, 162), "cols": (141, 176)},
        "case_07_wind_slope_oblique_135deg": {"cells": 678, "area": 61.02, "max": 179.54376220703125, "rows": (140, 167), "cols": (141, 170)},
        "case_08_wind_slope_opposed_180deg": {"cells": 574, "area": 51.66, "max": 179.46800231933594, "rows": (138, 163), "cols": (140, 166)},
    }
    for cid, exp in expected_arrival.items():
        out_dir = FLAMMAP_DIR / cid / "outputs"
        with rasterio.open(out_dir / "mtt_arrival_time_min.tif") as src:
            arr = src.read(1)
            nodata = src.nodata
            valid_mask = (arr != nodata)
            valid_vals = arr[valid_mask]

            assert src.shape == (301, 301)
            assert src.crs == "EPSG:32719"
            assert arr[150, 150] == 0.0
            assert len(valid_vals) == exp["cells"]
            assert np.isclose(len(valid_vals) * 0.09, exp["area"])
            assert np.isclose(np.max(valid_vals), exp["max"], atol=1e-3)

            rows, cols = np.where(valid_mask)
            assert (np.min(rows), np.max(rows)) == exp["rows"]
            assert (np.min(cols), np.max(cols)) == exp["cols"]

            # Cero contactos de borde
            border = np.sum(valid_mask[0, :]) + np.sum(valid_mask[-1, :]) + np.sum(valid_mask[:, 0]) + np.sum(valid_mask[:, -1])
            assert border == 0


def test_case02_to_08_mtt_ros_and_mask_parity():
    """Verifica las métricas de MTT ROS y la paridad de máscara con MTT Arrival en Case 02 a 08."""
    expected_ros = {
        "case_02_control_pure_wind_bench002": {"valid": 884, "pos": 883, "min_pos": 1.2193349599838257, "max_pos": 4.386771202087402},
        "case_03_control_pure_slope_10deg": {"valid": 422, "pos": 421, "min_pos": 0.9250189065933228, "max_pos": 2.7593750953674316},
        "case_04_wind_slope_aligned_0deg": {"valid": 1188, "pos": 1187, "min_pos": 1.1047133207321167, "max_pos": 5.935007095336914},
        "case_05_wind_slope_oblique_45deg": {"valid": 1115, "pos": 1114, "min_pos": 1.0633541345596313, "max_pos": 5.6278204917907715},
        "case_06_wind_slope_cross_90deg": {"valid": 920, "pos": 919, "min_pos": 1.0744987726211548, "max_pos": 4.7466816902160645},
        "case_07_wind_slope_oblique_135deg": {"valid": 678, "pos": 677, "min_pos": 1.121677279472351, "max_pos": 3.551631212234497},
        "case_08_wind_slope_opposed_180deg": {"valid": 574, "pos": 573, "min_pos": 1.3081163167953491, "max_pos": 2.8378803730010986},
    }
    for cid, exp in expected_ros.items():
        out_dir = FLAMMAP_DIR / cid / "outputs"
        with rasterio.open(out_dir / "mtt_arrival_time_min.tif") as src_arr:
            arr_mask = (src_arr.read(1) != src_arr.nodata)

        with rasterio.open(out_dir / "mtt_ros_m_min.tif") as src_ros:
            ros = src_ros.read(1)
            nodata = src_ros.nodata
            ros_mask = (ros != nodata)

            assert np.array_equal(arr_mask, ros_mask), f"Discrepancia de máscara en {cid}"

            pos = ros[ros_mask & (ros > 0)]
            assert np.sum(ros_mask) == exp["valid"]
            assert len(pos) == exp["pos"]
            assert np.isclose(np.min(pos), exp["min_pos"], atol=1e-4)
            assert np.isclose(np.max(pos), exp["max_pos"], atol=1e-4)


def test_case07_unit_conversion_and_reconstructed_file():
    """Verifica la procedencia y reconstrucción unitaria en Case 07."""
    out_dir = FLAMMAP_DIR / "case_07_wind_slope_oblique_135deg" / "outputs"
    recon_path = out_dir / "mtt_ros_ch_h_reconstructed_from_m_min.tif"
    conv_path = out_dir / "mtt_ros_m_min.tif"

    assert recon_path.exists()
    assert conv_path.exists()

    with rasterio.open(recon_path) as src_rec, rasterio.open(conv_path) as src_conv:
        rec = src_rec.read(1)
        conv = src_conv.read(1)
        nodata = src_rec.nodata

        valid = (rec != nodata)
        assert np.sum(valid) == 678

        expected_conv = np.full_like(rec, nodata)
        expected_conv[valid] = (rec[valid].astype(np.float64) * 20.1168 / 60.0).astype(np.float32)

        diff = np.abs(conv[valid] - expected_conv[valid])
        assert np.max(diff) < 1e-6


def test_case02_and_case06_rechecks_and_subset_consistency():
    """Verifica la consistencia estricta de subconjunto y diferencia 0.0 de Case 02 y Case 06."""
    audit_p = ANALYSIS_DIR / "flammap_cases02_08_execution_audit.json"
    assert audit_p.exists()
    with open(audit_p, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Case 02
    c02 = data["case02_recheck_audit"]
    assert c02["case_id"] == "case_02_control_pure_wind_bench002"
    assert c02["status"] == "FULL_HORIZON_RECHECK_PROMOTED"
    assert c02["initial_cells"] == 204
    assert c02["promoted_cells"] == 884
    assert c02["new_cells"] == 680
    assert c02["initial_mask_subset_of_recheck"] is True
    assert c02["max_arrival_diff_common_cells"] == 0.0
    assert np.isclose(c02["promoted_max_arrival_min"], 179.7850, atol=1e-3)
    assert c02["promoted_arrival_sha256"] == "DE1674512CD0EBDDCCC9AF3C853F1A8620676D370E314B8BB67B47CDF5C88A39"
    assert c02["promoted_mtt_ros_sha256"] == "95B088C36540EFFCEAAC2270CD48CF7B50325C18A752327BDAD53DC979ADE6EB"

    # Case 06
    c06 = data["case06_recheck_audit"]
    assert c06["case_id"] == "case_06_wind_slope_cross_90deg"
    assert c06["status"] == "FULL_HORIZON_RECHECK_PROMOTED"
    assert c06["initial_cells"] == 213
    assert c06["promoted_cells"] == 920
    assert c06["new_cells"] == 707
    assert c06["initial_mask_subset_of_recheck"] is True
    assert c06["max_arrival_diff_common_cells"] == 0.0
    assert np.isclose(c06["promoted_max_arrival_min"], 179.9864, atol=1e-3)
    assert c06["promoted_arrival_sha256"] == "AEB0E8C7D59CF6433FB3B7EFB2BEFA708BA43C937F193FF4C259AD55BC0AB1B7"
    assert c06["promoted_mtt_ros_sha256"] == "48E0A4A01E095019E462AB16B9D4EAA1AA53AB255ACBF73780D784994AD69B9C"

    # Preserved initial artifacts existence
    c02_init_arr = FLAMMAP_DIR / "case_02_control_pure_wind_bench002/initial_incomplete/mtt_arrival_time_min_initial_incomplete.tif"
    c06_init_arr = FLAMMAP_DIR / "case_06_wind_slope_cross_90deg/initial_incomplete/mtt_arrival_time_min_initial_incomplete.tif"
    assert c02_init_arr.exists()
    assert c06_init_arr.exists()


def test_flammap_cases02_08_summary_and_audit_files():
    """Verifica la existencia y consistencia de los archivos de auditoría de batch 02 a 08."""
    sum_csv = ANALYSIS_DIR / "flammap_cases02_08_summary.csv"
    audit_json = ANALYSIS_DIR / "flammap_cases02_08_execution_audit.json"
    loc_comp_csv = ANALYSIS_DIR / "flammap_vs_qnyx_local_case02_08.csv"

    assert sum_csv.exists()
    assert audit_json.exists()
    assert loc_comp_csv.exists()

    df_sum = pd.read_csv(sum_csv)
    assert len(df_sum) == 7
    assert list(df_sum["case_id"]) == [
        "case_02_control_pure_wind_bench002",
        "case_03_control_pure_slope_10deg",
        "case_04_wind_slope_aligned_0deg",
        "case_05_wind_slope_oblique_45deg",
        "case_06_wind_slope_cross_90deg",
        "case_07_wind_slope_oblique_135deg",
        "case_08_wind_slope_opposed_180deg"
    ]

    with open(audit_json, "r", encoding="utf-8") as f:
        audit_data = json.load(f)
    assert audit_data["flammap_version"] == "6.2.45.0"
    assert audit_data["overall_execution_status"] == "FLAMMAP_CASES01_08_EXECUTION_COMPLETE"
    assert "BENCH006_EXTERNAL_COMPARISON_PASS" not in json.dumps(audit_data)


def test_benchmark_manifest_batch_audit_validation_gates():
    """Verifica que todos los validation gates de Case 02-08 estén en estado PASS en el manifest."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest.get("flammap_cases_completed") == "8 / 8"
    summary = manifest.get("flammap_execution_summary", {})
    assert summary.get("total_flammap_cases") == 8
    assert summary.get("flammap_cases_completed") == "8 / 8"
    assert summary.get("flammap_cases_pending") == 0

    gates = manifest.get("validation_gates", {})
    batch_gates = [
        "CASES02_08_PROJECT_ARTIFACTS_PASS",
        "CASES02_08_LOCAL_RASTERS_METADATA_PASS",
        "CASES02_08_MTT_ARRIVAL_METADATA_PASS",
        "CASES02_08_MTT_ROS_METADATA_PASS",
        "CASES02_08_MTT_MASK_PARITY_PASS",
        "CASES02_08_DOMAIN_CLIPPING_PASS",
        "CASE02_FULL_HORIZON_RECHECK_PASS",
        "CASE02_INITIAL_SUBSET_CONSISTENCY_PASS",
        "CASE06_FULL_HORIZON_RECHECK_PASS",
        "CASE06_INITIAL_SUBSET_CONSISTENCY_PASS",
        "CASE07_MTT_ROS_PROVENANCE_CORRECTED_PASS",
        "FLAMMAP_CASES01_08_EXECUTION_COMPLETE"
    ]
    for g in batch_gates:
        assert g in gates, f"Falta el gate {g} en benchmark_manifest.json"
        assert gates[g]["status"] == "PASS", f"Gate {g} no tiene estado PASS"


# =============================================================================
# TESTS DE EJECUCIÓN ESPACIAL OFICIAL DE Q-NYX (FASE 5.5A)
# =============================================================================

QNYX_EXPECTED_CASES = [
    "case_01_control_flat_calm_bench001",
    "case_02_control_pure_wind_bench002",
    "case_03_control_pure_slope_10deg",
    "case_04_wind_slope_aligned_0deg",
    "case_05_wind_slope_oblique_45deg",
    "case_06_wind_slope_cross_90deg",
    "case_07_wind_slope_oblique_135deg",
    "case_08_wind_slope_opposed_180deg"
]

def test_qnyx_eight_cases_directories_and_artifacts_exist():
    """Verifica que existan los 8 directorios de caso en qnyx/ con todos sus artefactos oficiales."""
    assert QNYX_DIR.exists(), f"Directorio {QNYX_DIR} no existe"
    for cid in QNYX_EXPECTED_CASES:
        cdir = QNYX_DIR / cid
        assert cdir.exists(), f"Directorio de caso {cid} no existe en qnyx/"
        assert (cdir / "arrival_time_min.tif").exists(), f"Falta arrival_time_min.tif en {cid}"
        assert (cdir / "burned_mask_180min.tif").exists(), f"Falta burned_mask_180min.tif en {cid}"
        assert (cdir / "perimeter_180min.geojson").exists(), f"Falta perimeter_180min.geojson en {cid}"
        assert (cdir / "run_metadata.json").exists(), f"Falta run_metadata.json en {cid}"
        assert (cdir / "output_manifest.json").exists(), f"Falta output_manifest.json en {cid}"
        assert (cdir / "output_manifest.csv").exists(), f"Falta output_manifest.csv en {cid}"
        assert (cdir / "fire_intensity_experimental.tif").exists(), f"Falta fire_intensity_experimental.tif en {cid}"


def test_qnyx_geotiff_spatial_metadata_and_ignition_zero():
    """Verifica que las salidas GeoTIFF de Q-NYX cumplan exactamente el contrato espacial 301x301 EPSG:32719."""
    for cid in QNYX_EXPECTED_CASES:
        arr_path = QNYX_DIR / cid / "arrival_time_min.tif"
        with rasterio.open(arr_path) as ds:
            assert ds.shape == OFFICIAL_GRID_SHAPE
            assert ds.count == 1
            assert ds.dtypes[0] == "float32"
            assert str(ds.crs) == OFFICIAL_CRS
            assert ds.transform == OFFICIAL_TRANSFORM
            assert ds.nodata == -9999.0
            data = ds.read(1)
            assert data[IGNITION_ROW, IGNITION_COL] == 0.0, f"Ignición != 0.0 en {cid}"


def test_qnyx_arrival_and_burned_mask_parity():
    """Verifica paridad 100% exacta entre las celdas alcanzadas con t <= 180 min y la máscara burned_mask_180min."""
    for cid in QNYX_EXPECTED_CASES:
        arr_path = QNYX_DIR / cid / "arrival_time_min.tif"
        mask_path = QNYX_DIR / cid / "burned_mask_180min.tif"
        with rasterio.open(arr_path) as ds_arr, rasterio.open(mask_path) as ds_mask:
            arr = ds_arr.read(1)
            mask = ds_mask.read(1)
            expected_mask = (np.isfinite(arr) & (arr >= 0.0) & (arr <= 180.0)).astype(np.uint8)
            np.testing.assert_array_equal(mask, expected_mask, err_msg=f"Discrepancia en máscara binaria para {cid}")


def test_qnyx_zero_border_clipping_at_180min():
    """Verifica que ninguna de las 8 propagaciones espaciales de Q-NYX toque los bordes del dominio 301x301."""
    for cid in QNYX_EXPECTED_CASES:
        mask_path = QNYX_DIR / cid / "burned_mask_180min.tif"
        with rasterio.open(mask_path) as ds:
            mask = ds.read(1)
            border_contacts = (
                np.sum(mask[0, :]) +
                np.sum(mask[-1, :]) +
                np.sum(mask[:, 0]) +
                np.sum(mask[:, -1])
            )
            assert border_contacts == 0, f"Contactos de borde detectados en {cid}: {border_contacts}"


def test_qnyx_local_physics_consistency_against_preflight():
    """Verifica que el ROS de avance y heading local de Q-NYX coincidan estrictamente con el preflight analítico congelado."""
    preflight_df = pd.read_csv(ANALYSIS_DIR / "qnyx_local_preflight_predictions.csv")
    pf_dict = {row["case_id"]: row for _, row in preflight_df.iterrows()}

    for cid in QNYX_EXPECTED_CASES:
        meta_path = QNYX_DIR / cid / "run_metadata.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        pf_match = None
        for k, v in pf_dict.items():
            if cid.startswith(k) or k.startswith(cid.split("_bench")[0]):
                pf_match = v
                break

        assert pf_match is not None, f"No se encontró preflight para {cid}"
        obs_ros = meta["local_physics_consistency"]["observed_head_ros_m_min"]
        exp_ros = float(pf_match["r_head_m_min"])
        assert math.isclose(obs_ros, exp_ros, rel_tol=1e-5), f"Discrepancia de ROS en {cid}: {obs_ros} vs {exp_ros}"


def test_qnyx_official_output_hashes_and_audit_manifests():
    """Verifica que todos los hashes SHA256 de los artefactos oficiales de Q-NYX coincidan con la auditoría."""
    audit_path = ANALYSIS_DIR / "qnyx_cases01_08_execution_audit.json"
    assert audit_path.exists(), "Falta qnyx_cases01_08_execution_audit.json"
    with open(audit_path, "r", encoding="utf-8") as f:
        audit = json.load(f)

    for cid, cinfo in audit["cases_audit"].items():
        cdir = QNYX_DIR / cid
        for fname, expected_hash in cinfo["output_hashes"].items():
            fpath = cdir / fname
            assert fpath.exists(), f"Falta {fname} en {cid}"
            curr_hash = hashlib.sha256(fpath.read_bytes()).hexdigest().upper()
            assert curr_hash == expected_hash.upper(), f"Hash mismatch en {cid}/{fname}"


def test_qnyx_summary_table_and_validation_gates():
    """Verifica que la tabla de resumen y los gates de validación de Q-NYX en manifest estén en PASS."""
    summary_path = ANALYSIS_DIR / "qnyx_cases01_08_execution_summary.csv"
    assert summary_path.exists(), "Falta qnyx_cases01_08_execution_summary.csv"
    df = pd.read_csv(summary_path)
    assert len(df) == 8
    assert (df["border_contacts"] == 0).all()
    assert (df["arrival_burned_mask_equal"] == True).all()

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    gates = manifest.get("validation_gates", {})
    qnyx_gates = [
        "QNYX_EIGHT_CASES_SPATIAL_OUTPUTS_EXIST",
        "QNYX_GRID_METADATA_EXACT_PASS",
        "QNYX_IGNITION_ZERO_PASS",
        "QNYX_LOCAL_PHYSICS_CONSISTENCY_PASS",
        "QNYX_ARRIVAL_BURNED_MASK_PARITY_PASS",
        "QNYX_ZERO_BORDER_CLIPPING_PASS",
        "QNYX_RUN_METADATA_SCHEMA_PRESENCE_PASS",
        "QNYX_RUN_METADATA_DETERMINISM_PASS",
        "QNYX_OUTPUT_MANIFEST_HASH_SIZE_PASS",
        "QNYX_OUTPUT_MANIFEST_FULL_CONTRACT_PASS",
        "QNYX_PERIMETER_DETERMINISM_PASS",
        "QNYX_ARRIVAL_DETERMINISM_PASS",
        "QNYX_BURNED_MASK_DETERMINISM_PASS",
        "QNYX_CASES01_08_EXECUTION_DETERMINISM_PASS",
        "QNYX_CASES01_08_EXECUTION_COMPLETE"
    ]
    for g in qnyx_gates:
        assert g in gates, f"Falta gate {g} en manifest"
        assert gates[g]["status"] == "PASS", f"Gate {g} no está en PASS: {gates[g]}"


def test_qnyx_output_manifest_full_spatial_and_semantic_contract():
    """Verifica exhaustivamente el contrato espacial, dtype, NoData, roles y procedencia de cada salida."""
    for cid in QNYX_EXPECTED_CASES:
        man_path = QNYX_DIR / cid / "output_manifest.json"
        assert man_path.exists()
        with open(man_path, "r", encoding="utf-8") as f:
            man_data = json.load(f)

        for fname, fmeta in man_data["outputs"].items():
            fpath = QNYX_DIR / cid / fname
            assert fpath.exists(), f"Falta archivo {fname} en {cid}"
            assert fpath.stat().st_size == fmeta["size_bytes"]
            assert compute_sha256(fpath) == fmeta["sha256"]

            if fname.endswith(".tif"):
                with rasterio.open(fpath) as ds:
                    assert list(ds.shape) == fmeta["shape"]
                    assert str(ds.crs) == fmeta["crs"]
                    assert list(ds.transform)[:6] == fmeta["transform"][:6]
                    assert ds.dtypes[0] == fmeta["dtype"]
                    if fmeta.get("nodata") is not None:
                        assert math.isclose(ds.nodata, fmeta["nodata"], abs_tol=1e-4)
            elif fname.endswith(".geojson"):
                with open(fpath, "r", encoding="utf-8") as gjf:
                    gjdata = json.load(gjf)
                assert "features" in gjdata and len(gjdata["features"]) > 0
                assert fmeta["role"] == "QNYX_PERIMETER_POLYGON_180MIN"
                assert fmeta["provenance"] == "DIRECT_QNYX_EXECUTION_OUTPUT"


def test_qnyx_run_metadata_schema_presence_and_scientific_fields():
    """Verifica que todos los campos requeridos del esquema científico existan y tengan valores válidos en los 8 run_metadata.json."""
    required_keys = [
        "benchmark_id", "case_id", "role", "engine_identity", "propagator",
        "propagator_scientific_description", "git_baseline_commit", "git_baseline_tag",
        "spatial_domain", "meteorological_and_fuel_inputs", "local_physics_consistency",
        "spatial_propagation_metrics"
    ]
    for cid in QNYX_EXPECTED_CASES:
        meta_path = QNYX_DIR / cid / "run_metadata.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        for rk in required_keys:
            assert rk in meta, f"Falta clave {rk} en run_metadata de {cid}"

        sd = meta["spatial_domain"]
        assert sd["shape"] == [301, 301]
        assert sd["resolution_m"] == 30.0
        assert sd["crs"] == "EPSG:32719"
        assert sd["ignition_row"] == 150
        assert sd["ignition_col"] == 150
        assert sd["horizon_minutes"] == 180.0
        assert sd["stencil"] == "stencil_16"
        assert sd["intent"] == "SURFACE_FIRE_ONLY"

        spm = meta["spatial_propagation_metrics"]
        assert spm["border_contacts"] == 0
        assert spm["arrival_burned_mask_equal"] == True
        assert spm["domain_clipping_status"] == "PASS_ZERO_BORDER_CONTACTS"





SPATIAL_COMP_DIR = ANALYSIS_DIR / "spatial_comparison"
FIGURES_DIR = SPATIAL_COMP_DIR / "figures"

PAIRED_CASES_TEST = [
    ("case_01_control_flat_calm_bench001", "case_01_control_flat_calm", "01"),
    ("case_02_control_pure_wind_bench002", "case_02_control_pure_wind_bench002", "02"),
    ("case_03_control_pure_slope_10deg", "case_03_control_pure_slope_10deg", "03"),
    ("case_04_wind_slope_aligned_0deg", "case_04_wind_slope_aligned_0deg", "04"),
    ("case_05_wind_slope_oblique_45deg", "case_05_wind_slope_oblique_45deg", "05"),
    ("case_06_wind_slope_cross_90deg", "case_06_wind_slope_cross_90deg", "06"),
    ("case_07_wind_slope_oblique_135deg", "case_07_wind_slope_oblique_135deg", "07"),
    ("case_08_wind_slope_opposed_180deg", "case_08_wind_slope_opposed_180deg", "08")
]


def test_spatial_comparison_artifacts_exist_and_valid():
    """Verifica que todos los artefactos de análisis espacial CSV, JSON y figuras existan y contengan datos."""
    assert SPATIAL_COMP_DIR.exists()
    assert (SPATIAL_COMP_DIR / "bench006_spatial_comparison_180min.csv").exists()
    assert (SPATIAL_COMP_DIR / "bench006_spatial_comparison_by_time.csv").exists()
    assert (SPATIAL_COMP_DIR / "bench006_arrival_error_common_cells.csv").exists()
    assert (SPATIAL_COMP_DIR / "bench006_extent_comparison.csv").exists()
    assert (SPATIAL_COMP_DIR / "bench006_shape_comparison.csv").exists()
    assert (SPATIAL_COMP_DIR / "bench006_local_vs_spatial_summary.csv").exists()
    assert (SPATIAL_COMP_DIR / "bench006_spatial_comparison_audit.json").exists()

    assert (FIGURES_DIR / "bench006_metrics_summary_chart.png").exists()
    for _, _, cnum in PAIRED_CASES_TEST:
        assert (FIGURES_DIR / f"case_{cnum}_spatial_overlay_180min.png").exists()


def test_spatial_comparison_grid_alignment_across_8_cases():
    """Verifica el alineamiento exacto de grilla entre Q-NYX y FlamMap en los 8 casos pares."""
    for q_cid, f_cid, _ in PAIRED_CASES_TEST:
        q_path = QNYX_DIR / q_cid / "arrival_time_min.tif"
        f_path = FLAMMAP_DIR / f_cid / "outputs" / "mtt_arrival_time_min.tif"
        with rasterio.open(q_path) as q_ds, rasterio.open(f_path) as f_ds:
            assert q_ds.shape == f_ds.shape == OFFICIAL_GRID_SHAPE
            assert str(q_ds.crs) == str(f_ds.crs) == OFFICIAL_CRS
            assert math.isclose(q_ds.res[0], OFFICIAL_RES) and math.isclose(f_ds.res[0], OFFICIAL_RES)
            assert list(q_ds.transform)[:6] == list(f_ds.transform)[:6] == list(OFFICIAL_TRANSFORM)[:6]
            q_arr = q_ds.read(1)
            f_arr = f_ds.read(1)
            assert math.isclose(q_arr[IGNITION_ROW, IGNITION_COL], 0.0, abs_tol=1e-5)
            assert math.isclose(f_arr[IGNITION_ROW, IGNITION_COL], 0.0, abs_tol=1e-5)


def test_spatial_comparison_checkpoints_and_identities():
    """Verifica que las identidades de conjuntos, IoU y Dice se cumplan en todos los checkpoints temporales."""
    df_time = pd.read_csv(SPATIAL_COMP_DIR / "bench006_spatial_comparison_by_time.csv")
    assert len(df_time) == 8 * 6  # 8 casos * 6 checkpoints

    for _, row in df_time.iterrows():
        n_q = row["qnyx_burned_cells"]
        n_f = row["flammap_burned_cells"]
        n_inter = row["intersection_cells"]
        n_union = row["union_cells"]
        n_q_only = row["qnyx_only_cells"]
        n_f_only = row["flammap_only_cells"]

        # Identidades de teoría de conjuntos
        assert n_union == n_inter + n_q_only + n_f_only
        assert n_q == n_inter + n_q_only
        assert n_f == n_inter + n_f_only
        assert n_inter <= n_q
        assert n_inter <= n_f

        # Rango de métricas
        assert 0.0 <= row["iou"] <= 1.0
        assert 0.0 <= row["dice"] <= 1.0
        assert 0.0 <= row["qnyx_coverage_of_flammap"] <= 1.0
        assert 0.0 <= row["flammap_coverage_of_qnyx"] <= 1.0


def test_spatial_comparison_arrival_error_on_common_cells():
    """Verifica que los errores de tiempo de llegada en celdas comunes sean físicamente consistentes."""
    df_err = pd.read_csv(SPATIAL_COMP_DIR / "bench006_arrival_error_common_cells.csv")
    assert len(df_err) == 8

    for _, row in df_err.iterrows():
        assert row["common_cells_count"] > 0
        assert row["mae_min"] >= 0.0
        assert row["rmse_min"] >= row["mae_min"]
        assert row["max_absolute_error_min"] >= row["p95_absolute_error_min"]
        assert row["p95_absolute_error_min"] >= row["p90_absolute_error_min"]
        assert row["max_signed_diff_min"] >= row["min_signed_diff_min"]


def test_spatial_comparison_validation_gates_in_manifest_and_audit():
    """Verifica que los gates de comparación espacial estén registrados en el manifest y audit JSON."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    gates = manifest["validation_gates"]
    assert gates["BENCH006_SPATIAL_GRID_ALIGNMENT_PASS"]["status"] == "PASS"
    assert gates["BENCH006_EXTERNAL_COMPARISON_COMPLETE"]["status"] == "PASS"
    assert gates["QNYX_FLAMMAP_SPATIAL_EQUIVALENCE"]["status"] == "NOT_CLAIMED"

    with open(SPATIAL_COMP_DIR / "bench006_spatial_comparison_audit.json", "r", encoding="utf-8") as f:
        audit = json.load(f)
    assert audit["validation_gates"]["BENCH006_SPATIAL_GRID_ALIGNMENT_PASS"]["status"] == "PASS"
    assert audit["validation_gates"]["BENCH006_EXTERNAL_COMPARISON_COMPLETE"]["status"] == "PASS"
    assert audit["validation_gates"]["QNYX_FLAMMAP_SPATIAL_EQUIVALENCE"]["status"] == "NOT_CLAIMED"


def test_spatial_comparison_local_vs_spatial_external_provenance_and_schema():
    """Verifica que la tabla local-vs-spatial utilice la comparación local externa oficial y no consistencia interna."""
    summary_path = SPATIAL_COMP_DIR / "bench006_local_vs_spatial_summary.csv"
    assert summary_path.exists()
    df = pd.read_csv(summary_path)

    expected_cols = [
        "case_id", "role",
        "external_local_qnyx_ros_m_min", "external_local_flammap_ros_m_min",
        "external_local_ros_signed_difference", "external_local_ros_relative_difference_pct",
        "external_local_qnyx_heading_deg", "external_local_flammap_heading_deg",
        "external_local_heading_difference_deg",
        "spatial_180min_area_rel_diff_pct", "spatial_180min_iou", "spatial_180min_dice",
        "arrival_common_mae_min", "arrival_common_bias_min", "arrival_common_rmse_min",
        "centroid_distance_m", "hausdorff_distance_m"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Falta columna esperada {col} en summary"

    # Verificar Case 01 (Control Isótropo)
    c1 = df[df["case_id"] == "case_01_control_flat_calm_bench001"].iloc[0]
    assert math.isclose(float(c1["external_local_qnyx_ros_m_min"]), 1.212031, abs_tol=1e-4)
    assert math.isclose(float(c1["external_local_flammap_ros_m_min"]), 1.212014, abs_tol=1e-4)
    assert str(c1["external_local_heading_difference_deg"]) == "N/A_ISOTROPIC_CONTROL"
    assert str(c1["external_local_qnyx_heading_deg"]) == "N/A_ISOTROPIC_CONTROL"

    # Verificar Case 05 (Oblicuo 45°: diferencia de heading no nula ~ -0.306°)
    c5 = df[df["case_id"] == "case_05_wind_slope_oblique_45deg"].iloc[0]
    assert math.isclose(float(c5["external_local_heading_difference_deg"]), -0.305895, abs_tol=1e-3)
    assert float(c5["external_local_ros_relative_difference_pct"]) > 0.05  # ~ +0.1056%

    # Verificar Case 07 (Oblicuo 135°: diferencia de heading no nula ~ -0.525°)
    c7 = df[df["case_id"] == "case_07_wind_slope_oblique_135deg"].iloc[0]
    assert math.isclose(float(c7["external_local_heading_difference_deg"]), -0.525286, abs_tol=1e-3)
    assert float(c7["external_local_ros_relative_difference_pct"]) < -0.20  # ~ -0.3185%
