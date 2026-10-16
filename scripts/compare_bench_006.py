"""
scripts/compare_qnyx_flammap_bench_006.py

Script oficial de comparación espacial entre las salidas congeladas de Q-NYX
(propagador ráster Dijkstra de tiempo mínimo) y FlamMap (propagador MTT en retículo)
para los 8 casos oficiales de QNYX-BENCH-006.

Fase 5.6A — COMPARACIÓN ESPACIAL (ANALYSIS ONLY).
No ejecuta modelos, no altera físicas, no recalibra.
"""

import json
import math
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from shapely.geometry import Polygon, MultiPolygon, shape
from shapely.ops import unary_union
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Rutas principales
import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
ROOT_DIR = REPO_ROOT
BENCH_DIR = ROOT_DIR / "benchmarks" / "qnyx_bench_006_flammap_wind_slope"
QNYX_DIR = BENCH_DIR / "qnyx"
FLAMMAP_DIR = BENCH_DIR / "flammap"
ANALYSIS_DIR = BENCH_DIR / "analysis"
SPATIAL_COMP_DIR = ANALYSIS_DIR / "spatial_comparison"
FIGURES_DIR = SPATIAL_COMP_DIR / "figures"
PREFLIGHT_PATH = ANALYSIS_DIR / "qnyx_local_preflight_predictions.csv"
LOCAL_COMP_PATH = ANALYSIS_DIR / "flammap_vs_qnyx_local_case02_08.csv"

# Contrato de grilla oficial
OFFICIAL_GRID_SHAPE = (301, 301)
OFFICIAL_RES = 30.0
OFFICIAL_CRS = "EPSG:32719"
OFFICIAL_TRANSFORM = Affine(30.0, 0.0, 188850.0, 0.0, -30.0, 6077580.0)
IGNITION_ROW = 150
IGNITION_COL = 150
IGNITION_X = 193365.0
IGNITION_Y = 6073065.0

CHECKPOINTS_MIN = [30.0, 60.0, 90.0, 120.0, 150.0, 180.0]

CASES_PAIRED = [
    {
        "qnyx_case_id": "case_01_control_flat_calm_bench001",
        "flammap_case_id": "case_01_control_flat_calm",
        "case_num": "01",
        "role": "CONTROL_FLAT_CALM",
        "heading_deg": 0.0,
        "is_isotropic": True
    },
    {
        "qnyx_case_id": "case_02_control_pure_wind_bench002",
        "flammap_case_id": "case_02_control_pure_wind_bench002",
        "case_num": "02",
        "role": "CONTROL_PURE_WIND",
        "heading_deg": 90.0,
        "is_isotropic": False
    },
    {
        "qnyx_case_id": "case_03_control_pure_slope_10deg",
        "flammap_case_id": "case_03_control_pure_slope_10deg",
        "case_num": "03",
        "role": "CONTROL_PURE_SLOPE",
        "heading_deg": 90.0,
        "is_isotropic": False
    },
    {
        "qnyx_case_id": "case_04_wind_slope_aligned_0deg",
        "flammap_case_id": "case_04_wind_slope_aligned_0deg",
        "case_num": "04",
        "role": "WIND_SLOPE_ALIGNED",
        "heading_deg": 90.0,
        "is_isotropic": False
    },
    {
        "qnyx_case_id": "case_05_wind_slope_oblique_45deg",
        "flammap_case_id": "case_05_wind_slope_oblique_45deg",
        "case_num": "05",
        "role": "WIND_SLOPE_OBLIQUE_45",
        "heading_deg": 75.6122,
        "is_isotropic": False
    },
    {
        "qnyx_case_id": "case_06_wind_slope_cross_90deg",
        "flammap_case_id": "case_06_wind_slope_cross_90deg",
        "case_num": "06",
        "role": "WIND_SLOPE_CROSSWIND_90",
        "heading_deg": 63.9892,
        "is_isotropic": False
    },
    {
        "qnyx_case_id": "case_07_wind_slope_oblique_135deg",
        "flammap_case_id": "case_07_wind_slope_oblique_135deg",
        "case_num": "07",
        "role": "WIND_SLOPE_OBLIQUE_135",
        "heading_deg": 117.7811,
        "is_isotropic": False
    },
    {
        "qnyx_case_id": "case_08_wind_slope_opposed_180deg",
        "flammap_case_id": "case_08_wind_slope_opposed_180deg",
        "case_num": "08",
        "role": "WIND_SLOPE_OPPOSED_180",
        "heading_deg": 90.0,
        "is_isotropic": False
    }
]


def compute_sha256(file_path: Path) -> str:
    """Calcula hash SHA256 en mayúsculas."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()


def get_mask_polygon(binary_mask: np.ndarray, transform: Affine) -> Polygon:
    """Genera un polígono Shapely unificado a partir de las celdas de una máscara binaria."""
    rows, cols = np.where(binary_mask)
    if len(rows) == 0:
        return Polygon()
    
    cell_polys = []
    for r, c in zip(rows, cols):
        x_min, y_max = transform * (c, r)
        x_max, y_min = transform * (c + 1, r + 1)
        poly = Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])
        cell_polys.append(poly)
    
    merged = unary_union(cell_polys)
    return merged


def write_analysis_geotiff(path: Path, data: np.ndarray, dtype: str, nodata_val, profile: dict):
    """Escribe un archivo GeoTIFF de análisis derivado."""
    prof = profile.copy()
    prof.update({
        "dtype": dtype,
        "nodata": nodata_val,
        "count": 1
    })
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(data.astype(dtype), 1)


def run_spatial_comparison():
    print("===============================================================================")
    print("=== INICIANDO COMPARACIÓN ESPACIAL FORMAL: Q-NYX DIJKSTRA vs FLAMMAP MTT ===")
    print("===============================================================================\n")

    SPATIAL_COMP_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Cargar datos preflight y local comparison
    preflight_df = pd.read_csv(PREFLIGHT_PATH)
    pf_dict = {row["case_id"]: row for _, row in preflight_df.iterrows()}

    local_comp_df = pd.read_csv(LOCAL_COMP_PATH) if LOCAL_COMP_PATH.exists() else None

    # Contenedores de resultados
    comparison_180_rows = []
    comparison_by_time_rows = []
    arrival_error_rows = []
    extent_rows = []
    shape_rows = []
    local_vs_spatial_rows = []

    grid_alignment_all_pass = True

    for p in CASES_PAIRED:
        q_cid = p["qnyx_case_id"]
        f_cid = p["flammap_case_id"]
        c_num = p["case_num"]
        role = p["role"]
        heading_deg = p["heading_deg"]
        is_iso = p["is_isotropic"]

        print(f"--- Procesando Caso {c_num}: {role} ---")

        q_arr_path = QNYX_DIR / q_cid / "arrival_time_min.tif"
        f_arr_path = FLAMMAP_DIR / f_cid / "outputs" / "mtt_arrival_time_min.tif"

        assert q_arr_path.exists(), f"Falta ráster Q-NYX: {q_arr_path}"
        assert f_arr_path.exists(), f"Falta ráster FlamMap: {f_arr_path}"

        # 2. Verificación de alineamiento de grilla
        with rasterio.open(q_arr_path) as q_ds, rasterio.open(f_arr_path) as f_ds:
            q_profile = q_ds.profile
            q_arr = q_ds.read(1)
            f_arr = f_ds.read(1)

            # Verificar grilla
            shape_match = (q_ds.shape == f_ds.shape == OFFICIAL_GRID_SHAPE)
            crs_match = (str(q_ds.crs) == str(f_ds.crs) == OFFICIAL_CRS)
            res_match = (math.isclose(q_ds.res[0], OFFICIAL_RES) and math.isclose(f_ds.res[0], OFFICIAL_RES))
            trans_match = (list(q_ds.transform)[:6] == list(f_ds.transform)[:6] == list(OFFICIAL_TRANSFORM)[:6])
            ign_q_zero = math.isclose(q_arr[IGNITION_ROW, IGNITION_COL], 0.0, abs_tol=1e-5)
            ign_f_zero = math.isclose(f_arr[IGNITION_ROW, IGNITION_COL], 0.0, abs_tol=1e-5)

            grid_pass = shape_match and crs_match and res_match and trans_match and ign_q_zero and ign_f_zero
            if not grid_pass:
                grid_alignment_all_pass = False
                print(f"  [ERROR] Falla de alineamiento de grilla en Caso {c_num}!")

        # Valid mask logic
        q_valid_all = (q_arr != -9999.0) & np.isfinite(q_arr) & (q_arr >= 0.0)
        f_valid_all = (f_arr != -9999.0) & np.isfinite(f_arr) & (f_arr >= 0.0)

        # 3. Comparaciones temporales a 30, 60, 90, 120, 150, 180 min
        time_series_case = {}
        for t in CHECKPOINTS_MIN:
            q_burned_t = q_valid_all & (q_arr <= t)
            f_burned_t = f_valid_all & (f_arr <= t)

            n_q = int(np.sum(q_burned_t))
            n_f = int(np.sum(f_burned_t))
            area_q = float(n_q * 0.09)
            area_f = float(n_f * 0.09)

            inter_mask = q_burned_t & f_burned_t
            union_mask = q_burned_t | f_burned_t
            q_only_mask = q_burned_t & (~f_burned_t)
            f_only_mask = f_burned_t & (~q_burned_t)

            n_inter = int(np.sum(inter_mask))
            n_union = int(np.sum(union_mask))
            n_q_only = int(np.sum(q_only_mask))
            n_f_only = int(np.sum(f_only_mask))

            iou = float(n_inter / n_union) if n_union > 0 else 1.0
            dice = float(2.0 * n_inter / (n_q + n_f)) if (n_q + n_f) > 0 else 1.0
            cov_q_of_f = float(n_inter / n_f) if n_f > 0 else 1.0
            cov_f_of_q = float(n_inter / n_q) if n_q > 0 else 1.0

            area_diff_signed = float(area_q - area_f)
            area_diff_rel_pct = float((area_diff_signed / area_f) * 100.0) if area_f > 0 else 0.0
            area_ratio = float(area_q / area_f) if area_f > 0 else 1.0

            time_row = {
                "case_id": q_cid,
                "case_num": c_num,
                "role": role,
                "elapsed_minutes": t,
                "qnyx_burned_cells": n_q,
                "flammap_burned_cells": n_f,
                "qnyx_area_ha": area_q,
                "flammap_area_ha": area_f,
                "intersection_cells": n_inter,
                "union_cells": n_union,
                "qnyx_only_cells": n_q_only,
                "flammap_only_cells": n_f_only,
                "iou": iou,
                "dice": dice,
                "qnyx_coverage_of_flammap": cov_q_of_f,
                "flammap_coverage_of_qnyx": cov_f_of_q,
                "signed_area_diff_ha": area_diff_signed,
                "relative_area_diff_pct": area_diff_rel_pct,
                "area_ratio": area_ratio
            }
            comparison_by_time_rows.append(time_row)
            time_series_case[t] = time_row

        # Guardar resumen final de 180 min
        t180 = time_series_case[180.0]
        comparison_180_rows.append(t180)

        # 4. Error de tiempo de llegada en celdas comunes a 180 min
        q_180 = q_valid_all & (q_arr <= 180.0)
        f_180 = f_valid_all & (f_arr <= 180.0)
        common_180 = q_180 & f_180
        n_common = int(np.sum(common_180))

        if n_common > 0:
            diff_arrival = q_arr[common_180] - f_arr[common_180]
            abs_diff = np.abs(diff_arrival)

            bias = float(np.mean(diff_arrival))
            mae = float(np.mean(abs_diff))
            rmse = float(np.sqrt(np.mean(diff_arrival ** 2)))
            med_signed = float(np.median(diff_arrival))
            med_abs = float(np.median(abs_diff))
            p90_abs = float(np.percentile(abs_diff, 90))
            p95_abs = float(np.percentile(abs_diff, 95))
            max_abs = float(np.max(abs_diff))
            min_signed = float(np.min(diff_arrival))
            max_signed = float(np.max(diff_arrival))
        else:
            bias = mae = rmse = med_signed = med_abs = p90_abs = p95_abs = max_abs = min_signed = max_signed = 0.0

        arrival_err_row = {
            "case_id": q_cid,
            "case_num": c_num,
            "role": role,
            "common_cells_count": n_common,
            "common_area_ha": float(n_common * 0.09),
            "mean_signed_error_bias_min": bias,
            "mae_min": mae,
            "rmse_min": rmse,
            "median_signed_diff_min": med_signed,
            "median_absolute_error_min": med_abs,
            "p90_absolute_error_min": p90_abs,
            "p95_absolute_error_min": p95_abs,
            "max_absolute_error_min": max_abs,
            "min_signed_diff_min": min_signed,
            "max_signed_diff_min": max_signed
        }
        arrival_error_rows.append(arrival_err_row)

        # 5. Métricas de extensión espacial a 180 min
        q_rows, q_cols = np.where(q_180)
        f_rows, f_cols = np.where(f_180)

        q_r_min, q_r_max = int(np.min(q_rows)), int(np.max(q_rows))
        q_c_min, q_c_max = int(np.min(q_cols)), int(np.max(q_cols))
        f_r_min, f_r_max = int(np.min(f_rows)), int(np.max(f_rows))
        f_c_min, f_c_max = int(np.min(f_cols)), int(np.max(f_cols))

        q_north = float((IGNITION_ROW - q_r_min) * OFFICIAL_RES)
        q_south = float((q_r_max - IGNITION_ROW) * OFFICIAL_RES)
        q_east = float((q_c_max - IGNITION_COL) * OFFICIAL_RES)
        q_west = float((IGNITION_COL - q_c_min) * OFFICIAL_RES)

        f_north = float((IGNITION_ROW - f_r_min) * OFFICIAL_RES)
        f_south = float((f_r_max - IGNITION_ROW) * OFFICIAL_RES)
        f_east = float((f_c_max - IGNITION_COL) * OFFICIAL_RES)
        f_west = float((IGNITION_COL - f_c_min) * OFFICIAL_RES)

        # Coordenadas UTM centro de celdas
        q_x_cells = OFFICIAL_TRANSFORM[2] + (q_cols + 0.5) * OFFICIAL_RES
        q_y_cells = OFFICIAL_TRANSFORM[5] - (q_rows + 0.5) * OFFICIAL_RES
        f_x_cells = OFFICIAL_TRANSFORM[2] + (f_cols + 0.5) * OFFICIAL_RES
        f_y_cells = OFFICIAL_TRANSFORM[5] - (f_rows + 0.5) * OFFICIAL_RES

        q_cent_x = float(np.mean(q_x_cells))
        q_cent_y = float(np.mean(q_y_cells))
        f_cent_x = float(np.mean(f_x_cells))
        f_cent_y = float(np.mean(f_y_cells))

        cent_dist = float(math.sqrt((q_cent_x - f_cent_x) ** 2 + (q_cent_y - f_cent_y) ** 2))

        # 6. Extensión sobre el eje resultante
        rad_head = math.radians(heading_deg)
        u_long = np.array([math.sin(rad_head), math.cos(rad_head)])   # (dx, dy)
        u_trans = np.array([math.cos(rad_head), -math.sin(rad_head)]) # (+90° derecha)

        # Desplazamientos desde ignición (X_ign, Y_ign)
        q_disp = np.stack([q_x_cells - IGNITION_X, q_y_cells - IGNITION_Y], axis=1)
        f_disp = np.stack([f_x_cells - IGNITION_X, f_y_cells - IGNITION_Y], axis=1)

        q_proj_long = np.dot(q_disp, u_long)
        q_proj_trans = np.dot(q_disp, u_trans)
        f_proj_long = np.dot(f_disp, u_long)
        f_proj_trans = np.dot(f_disp, u_trans)

        q_head_ext = float(np.max(q_proj_long))
        q_back_ext = float(-np.min(q_proj_long))
        q_long_span = float(q_head_ext + q_back_ext)
        q_r_flank = float(np.max(q_proj_trans))
        q_l_flank = float(-np.min(q_proj_trans))
        q_trans_span = float(q_r_flank + q_l_flank)

        f_head_ext = float(np.max(f_proj_long))
        f_back_ext = float(-np.min(f_proj_long))
        f_long_span = float(f_head_ext + f_back_ext)
        f_r_flank = float(np.max(f_proj_trans))
        f_l_flank = float(-np.min(f_proj_trans))
        f_trans_span = float(f_r_flank + f_l_flank)

        extent_row = {
            "case_id": q_cid,
            "case_num": c_num,
            "role": role,
            "resultant_heading_deg": heading_deg,
            "qnyx_north_extent_m": q_north,
            "flammap_north_extent_m": f_north,
            "qnyx_south_extent_m": q_south,
            "flammap_south_extent_m": f_south,
            "qnyx_east_extent_m": q_east,
            "flammap_east_extent_m": f_east,
            "qnyx_west_extent_m": q_west,
            "flammap_west_extent_m": f_west,
            "qnyx_ns_span_m": float(q_north + q_south),
            "flammap_ns_span_m": float(f_north + f_south),
            "qnyx_ew_span_m": float(q_east + q_west),
            "flammap_ew_span_m": float(f_east + f_west),
            "qnyx_centroid_x": q_cent_x,
            "qnyx_centroid_y": q_cent_y,
            "flammap_centroid_x": f_cent_x,
            "flammap_centroid_y": f_cent_y,
            "centroid_distance_m": cent_dist,
            "qnyx_head_extent_m": q_head_ext,
            "flammap_head_extent_m": f_head_ext,
            "qnyx_back_extent_m": q_back_ext,
            "flammap_back_extent_m": f_back_ext,
            "qnyx_longitudinal_span_m": q_long_span,
            "flammap_longitudinal_span_m": f_long_span,
            "qnyx_right_flank_extent_m": q_r_flank,
            "flammap_right_flank_extent_m": f_r_flank,
            "qnyx_left_flank_extent_m": q_l_flank,
            "flammap_left_flank_extent_m": f_l_flank,
            "qnyx_transverse_span_m": q_trans_span,
            "flammap_transverse_span_m": f_trans_span
        }
        extent_rows.append(extent_row)

        # 7. Polígonos de forma con método unificado
        q_poly = get_mask_polygon(q_180, OFFICIAL_TRANSFORM)
        f_poly = get_mask_polygon(f_180, OFFICIAL_TRANSFORM)

        q_poly_area = float(q_poly.area) if not q_poly.is_empty else 0.0
        f_poly_area = float(f_poly.area) if not f_poly.is_empty else 0.0
        q_poly_len = float(q_poly.length) if not q_poly.is_empty else 0.0
        f_poly_len = float(f_poly.length) if not f_poly.is_empty else 0.0

        q_compactness = float((4.0 * math.pi * q_poly_area) / (q_poly_len ** 2)) if q_poly_len > 0 else 0.0
        f_compactness = float((4.0 * math.pi * f_poly_area) / (f_poly_len ** 2)) if f_poly_len > 0 else 0.0

        hausdorff_dist = float(q_poly.hausdorff_distance(f_poly)) if (not q_poly.is_empty and not f_poly.is_empty) else 0.0

        shape_row = {
            "case_id": q_cid,
            "case_num": c_num,
            "role": role,
            "qnyx_polygon_area_m2": q_poly_area,
            "flammap_polygon_area_m2": f_poly_area,
            "qnyx_perimeter_length_m": q_poly_len,
            "flammap_perimeter_length_m": f_poly_len,
            "qnyx_compactness": q_compactness,
            "flammap_compactness": f_compactness,
            "hausdorff_distance_m": hausdorff_dist
        }
        shape_rows.append(shape_row)

        # 8. Local vs Spatial Summary Table (External Comparison: Q-NYX vs FlamMap)
        if c_num == "01":
            ext_q_ros = 1.21203124935
            ext_f_ros = 1.2120143175125122
            ext_ros_diff_signed = float(ext_f_ros - ext_q_ros)
            ext_ros_diff_rel = float((ext_ros_diff_signed / ext_q_ros) * 100.0)
            ext_q_head = "N/A_ISOTROPIC_CONTROL"
            ext_f_head = "N/A_ISOTROPIC_CONTROL"
            ext_head_diff = "N/A_ISOTROPIC_CONTROL"
        else:
            loc_match = None
            if local_comp_df is not None:
                matches = local_comp_df[local_comp_df["case_id"].str.contains(f"case_{c_num}", case=False, na=False)]
                if len(matches) > 0:
                    loc_match = matches.iloc[0]

            if loc_match is not None:
                ext_q_ros = float(loc_match["qnyx_frozen_r_head_m_min"])
                ext_f_ros = float(loc_match["flammap_basic_ros_m_min"])
                ext_ros_diff_signed = float(loc_match["signed_ros_diff_m_min"])
                ext_ros_diff_rel = float(loc_match["relative_ros_diff_percent"])
                ext_q_head = float(loc_match["qnyx_frozen_heading_deg"])
                ext_f_head = float(loc_match["flammap_direction_deg"])
                ext_head_diff = float(loc_match["signed_circular_heading_diff_deg"])
            else:
                ext_q_ros = ext_f_ros = ext_ros_diff_signed = ext_ros_diff_rel = 0.0
                ext_q_head = ext_f_head = ext_head_diff = 0.0

        local_vs_spatial_row = {
            "case_id": q_cid,
            "role": role,
            "external_local_qnyx_ros_m_min": ext_q_ros,
            "external_local_flammap_ros_m_min": ext_f_ros,
            "external_local_ros_signed_difference": ext_ros_diff_signed,
            "external_local_ros_relative_difference_pct": ext_ros_diff_rel,
            "external_local_qnyx_heading_deg": ext_q_head,
            "external_local_flammap_heading_deg": ext_f_head,
            "external_local_heading_difference_deg": ext_head_diff,
            "spatial_180min_area_rel_diff_pct": t180["relative_area_diff_pct"],
            "spatial_180min_iou": t180["iou"],
            "spatial_180min_dice": t180["dice"],
            "arrival_common_mae_min": mae,
            "arrival_common_bias_min": bias,
            "arrival_common_rmse_min": rmse,
            "centroid_distance_m": cent_dist,
            "hausdorff_distance_m": hausdorff_dist
        }
        local_vs_spatial_rows.append(local_vs_spatial_row)

        # 9. Generar rasters derivados de análisis
        case_derived_dir = SPATIAL_COMP_DIR / q_cid
        case_derived_dir.mkdir(parents=True, exist_ok=True)

        class_arr = np.zeros(OFFICIAL_GRID_SHAPE, dtype=np.uint8)
        class_arr[common_180] = 1
        class_arr[q_180 & (~f_180)] = 2
        class_arr[f_180 & (~q_180)] = 3

        write_analysis_geotiff(
            case_derived_dir / "comparison_class_180min.tif",
            class_arr,
            "uint8",
            0,
            q_profile
        )

        diff_grid = np.full(OFFICIAL_GRID_SHAPE, -9999.0, dtype=np.float32)
        diff_grid[common_180] = q_arr[common_180] - f_arr[common_180]

        write_analysis_geotiff(
            case_derived_dir / "arrival_difference_common_min.tif",
            diff_grid,
            "float32",
            -9999.0,
            q_profile
        )

        # 10. Generar visualización por caso
        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
        
        cmap = matplotlib.colors.ListedColormap(['#f8f9fa', '#2b8a3e', '#1971c2', '#e8590c'])
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

        extent_utm = [
            OFFICIAL_TRANSFORM[2],
            OFFICIAL_TRANSFORM[2] + OFFICIAL_GRID_SHAPE[1] * OFFICIAL_RES,
            OFFICIAL_TRANSFORM[5] - OFFICIAL_GRID_SHAPE[0] * OFFICIAL_RES,
            OFFICIAL_TRANSFORM[5]
        ]

        im = ax.imshow(class_arr, cmap=cmap, norm=norm, extent=extent_utm, origin='upper')
        ax.plot(IGNITION_X, IGNITION_Y, marker='*', color='red', markersize=12, label='Ignición (150, 150)')

        all_r, all_c = np.where(q_180 | f_180)
        if len(all_r) > 0:
            min_c, max_c = max(0, np.min(all_c) - 15), min(300, np.max(all_c) + 15)
            min_r, max_r = max(0, np.min(all_r) - 15), min(300, np.max(all_r) + 15)
            x_min_v, y_max_v = OFFICIAL_TRANSFORM * (min_c, min_r)
            x_max_v, y_min_v = OFFICIAL_TRANSFORM * (max_c, max_r)
            ax.set_xlim(x_min_v, x_max_v)
            ax.set_ylim(y_min_v, y_max_v)

        ax.set_title(f"QNYX-BENCH-006: Caso {c_num} ({role})\nQ-NYX Dijkstra vs FlamMap MTT (180 min)", fontsize=11, fontweight='bold')
        ax.set_xlabel("UTM Este [m]", fontsize=9)
        ax.set_ylabel("UTM Norte [m]", fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)

        patches = [
            mpatches.Patch(color='#2b8a3e', label=f'Intersección (Ambos): {t180["intersection_cells"]} cel ({t180["intersection_cells"]*0.09:.1f} ha)'),
            mpatches.Patch(color='#1971c2', label=f'Q-NYX Only: {t180["qnyx_only_cells"]} cel ({t180["qnyx_only_cells"]*0.09:.1f} ha)'),
            mpatches.Patch(color='#e8590c', label=f'FlamMap Only: {t180["flammap_only_cells"]} cel ({t180["flammap_only_cells"]*0.09:.1f} ha)'),
            mpatches.Patch(color='none', label=f'IoU: {t180["iou"]:.3f} | Dice: {t180["dice"]:.3f} | Diff: {t180["relative_area_diff_pct"]:+.1f}%')
        ]
        ax.legend(handles=patches, loc='upper left', fontsize=8, framealpha=0.9)

        fig_path = FIGURES_DIR / f"case_{c_num}_spatial_overlay_180min.png"
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close(fig)

    # 11. Guardar tablas CSV oficiales
    df_180 = pd.DataFrame(comparison_180_rows)
    df_by_time = pd.DataFrame(comparison_by_time_rows)
    df_arr_err = pd.DataFrame(arrival_error_rows)
    df_extent = pd.DataFrame(extent_rows)
    df_shape = pd.DataFrame(shape_rows)
    df_local_vs_spatial = pd.DataFrame(local_vs_spatial_rows)

    df_180.to_csv(SPATIAL_COMP_DIR / "bench006_spatial_comparison_180min.csv", index=False)
    df_by_time.to_csv(SPATIAL_COMP_DIR / "bench006_spatial_comparison_by_time.csv", index=False)
    df_arr_err.to_csv(SPATIAL_COMP_DIR / "bench006_arrival_error_common_cells.csv", index=False)
    df_extent.to_csv(SPATIAL_COMP_DIR / "bench006_extent_comparison.csv", index=False)
    df_shape.to_csv(SPATIAL_COMP_DIR / "bench006_shape_comparison.csv", index=False)
    df_local_vs_spatial.to_csv(SPATIAL_COMP_DIR / "bench006_local_vs_spatial_summary.csv", index=False)

    print("\n[OK] Tablas CSV de análisis espacial generadas en analysis/spatial_comparison/.")

    # 12. Generar gráfico resumen
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    cases_labels = [f"C{r['case_num']}\n{r['role']}" for r in comparison_180_rows]
    ious = [r["iou"] for r in comparison_180_rows]
    dices = [r["dice"] for r in comparison_180_rows]
    area_diffs_pct = [r["relative_area_diff_pct"] for r in comparison_180_rows]

    x = np.arange(len(cases_labels))
    w = 0.35

    ax1.bar(x - w/2, ious, w, label='IoU', color='#1971c2')
    ax1.bar(x + w/2, dices, w, label='Dice', color='#2b8a3e')
    ax1.set_xticks(x)
    ax1.set_xticklabels(cases_labels, fontsize=7, rotation=30, ha='right')
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Métrica de Solapamiento [0, 1]", fontsize=9)
    ax1.set_title("Solapamiento Espacial Q-NYX vs FlamMap (180 min)", fontsize=10, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower right', fontsize=8)

    bars = ax2.bar(x, area_diffs_pct, color=['#e8590c' if v < 0 else '#2b8a3e' for v in area_diffs_pct])
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax2.set_xticks(x)
    ax2.set_xticklabels(cases_labels, fontsize=7, rotation=30, ha='right')
    ax2.set_ylabel("Diferencia Relativa de Área (%) [(Q - F)/F * 100]", fontsize=9)
    ax2.set_title("Diferencia de Huella Espacial a 180 min (%)", fontsize=10, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        ax2.annotate(f'{height:+.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -3),
                    textcoords="offset points",
                    ha='center', va=va, fontsize=7, fontweight='bold')

    plt.tight_layout()
    chart_path = FIGURES_DIR / "bench006_metrics_summary_chart.png"
    plt.savefig(chart_path)
    plt.close(fig)
    print(f"[OK] Gráfico resumen generado en: {chart_path.name}")

    # 13. Generar JSON de auditoría formal
    subset_characterization = [
        {
            "case_id": r["case_id"],
            "role": r["role"],
            "qnyx_burned_cells": r["qnyx_burned_cells"],
            "flammap_burned_cells": r["flammap_burned_cells"],
            "intersection_cells": r["intersection_cells"],
            "union_cells": r["union_cells"],
            "qnyx_only_cells": r["qnyx_only_cells"],
            "flammap_only_cells": r["flammap_only_cells"],
            "is_qnyx_strict_subset_of_flammap": (r["qnyx_only_cells"] == 0),
            "qnyx_fraction_in_flammap": float(r["intersection_cells"] / r["qnyx_burned_cells"]) if r["qnyx_burned_cells"] > 0 else 1.0
        }
        for r in comparison_180_rows
    ]

    audit_data = {
        "benchmark_id": "QNYX-BENCH-006",
        "phase": "5.6B",
        "comparison_title": "Formal Spatial Propagation Comparison: Q-NYX Raster Dijkstra vs FlamMap MTT",
        "comparison_role": "EXTERNAL_REFERENCE_COMPARISON_NOT_GROUND_TRUTH",
        "local_physics_comparison_distinction": {
            "INTERNAL_QNYX_LOCAL_CONSISTENCY": "Verificación interna del motor Q-NYX contra las predicciones analíticas de preflight (Gate QNYX_LOCAL_PHYSICS_CONSISTENCY_PASS = PASS).",
            "EXTERNAL_QNYX_FLAMMAP_LOCAL_COMPARISON": "Comparación directa entre salidas locales calculadas por Q-NYX vs calculadas por FlamMap GUI oficial (autoritativo: analysis/flammap_vs_qnyx_local_case02_08.csv y auditoría Case01)."
        },
        "grid_alignment_gate": "BENCH006_SPATIAL_GRID_ALIGNMENT_PASS",
        "grid_alignment_status": "PASS" if grid_alignment_all_pass else "FAIL",
        "checkpoints_minutes": CHECKPOINTS_MIN,
        "total_cases_analyzed": len(CASES_PAIRED),
        "validation_gates": {
            "BENCH006_SPATIAL_GRID_ALIGNMENT_PASS": {
                "status": "PASS" if grid_alignment_all_pass else "FAIL",
                "description": "Alineamiento geométrico exacto verificado entre las 8 salidas Q-NYX y las 8 salidas FlamMap (301x301 a 30m EPSG:32719 con ignición idéntica)."
            },
            "BENCH006_EXTERNAL_COMPARISON_COMPLETE": {
                "status": "PASS",
                "description": "Comparación espacial exhaustiva multi-horizonte completada en los 8 casos oficiales."
            },
            "QNYX_FLAMMAP_SPATIAL_EQUIVALENCE": {
                "status": "NOT_CLAIMED",
                "description": "No se define ni reclama equivalencia espacial absoluta debido a diferencias estructurales de propagación entre ráster Dijkstra y MTT."
            }
        },
        "subset_characterization_180min": subset_characterization,
        "spatial_comparison_summary_180min": comparison_180_rows,
        "arrival_error_summary_common_cells": arrival_error_rows,
        "extent_summary": extent_rows,
        "shape_summary": shape_rows,
        "local_vs_spatial_summary": local_vs_spatial_rows
    }

    audit_json_path = SPATIAL_COMP_DIR / "bench006_spatial_comparison_audit.json"
    with open(audit_json_path, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=4)

    print(f"[OK] JSON de auditoría formal generado en: {audit_json_path.name}")
    print("\n===============================================================================")
    print("=== COMPARACIÓN ESPACIAL DE QNYX-BENCH-006 COMPLETADA CON ÉXITO ===")
    print("===============================================================================\n")


if __name__ == "__main__":
    run_spatial_comparison()
