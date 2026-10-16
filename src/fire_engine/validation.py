"""
validation.py

Validación previa de escenarios e insumos de simulación.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
from fire_engine.schemas import ScenarioConfig
from fire_engine.raster_loader import LandscapeData

class ScenarioValidationError(ValueError):
    pass

def validate_scenario(
    scenario: ScenarioConfig,
    base_dir: Path,
    landscape: LandscapeData
) -> Tuple[bool, int, int, str]:
    """
    Valida las condiciones de inicio de la simulación.
    Retorna (is_valid, row, col, message).
    """
    ign = scenario.ignition

    # 1. Validar Duración e Intervalo
    if scenario.duration_minutes <= 0:
        raise ScenarioValidationError(f"Duración inválida ({scenario.duration_minutes} min), debe ser > 0.")
    if scenario.output_interval_minutes <= 0:
        raise ScenarioValidationError(f"Intervalo de salida inválido ({scenario.output_interval_minutes} min), debe ser > 0.")

    # 2. Determinar Fila / Columna de Ignición
    row, col = None, None
    if ign.row is not None and ign.col is not None:
        row, col = ign.row, ign.col
    elif ign.lon is not None and ign.lat is not None:
        row, col = landscape.domain.wgs84_to_grid(ign.lon, ign.lat)

    if row is None or col is None:
        raise ScenarioValidationError("No se especificaron coordenadas de ignición válidas (lon/lat o row/col).")

    # 3. Validar si está dentro del dominio
    if not landscape.domain.is_inside(row, col):
        err_msg = f"Ignición ({row}, {col}) fuera de los límites del dominio ({landscape.domain.rows}x{landscape.domain.cols})."
        create_corrected_scenario(scenario, base_dir, 500, 500, err_msg)
        raise ScenarioValidationError(err_msg)

    # 4. Validar celda combustible
    fuel_code = landscape.fuel_model[row, col]
    if fuel_code == 0 or np_isnan(fuel_code):
        err_msg = f"Ignición ({row}, {col}) está en celda sin combustible (código={fuel_code})."
        # Buscar celda combustible cercana
        alt_r, alt_c = find_nearest_combustible_cell(landscape, row, col)
        create_corrected_scenario(scenario, base_dir, alt_r, alt_c, err_msg)
        raise ScenarioValidationError(err_msg)

    # 5. Validar celda barrera absoluta / agua
    barrier_code = landscape.barriers[row, col]
    if barrier_code in (1, 5):  # 1=agua, 5=suelo desnudo
        err_msg = f"Ignición ({row}, {col}) está sobre cuerpo de agua o barrera absoluta (código={barrier_code})."
        alt_r, alt_c = find_nearest_combustible_cell(landscape, row, col)
        create_corrected_scenario(scenario, base_dir, alt_r, alt_c, err_msg)
        raise ScenarioValidationError(err_msg)

    # 6. Validaciones de fuel_moisture
    if scenario.fuel_moisture is not None:
        if scenario.fuel_moisture.mode not in ("constant", "hourly_dynamic", "external"):
            raise ScenarioValidationError(
                f"Modo de fuel_moisture no válido: '{scenario.fuel_moisture.mode}'. "
                "Modos permitidos: 'constant', 'hourly_dynamic', 'external'."
            )

    # 7. Validaciones específicas para kernel Rothermel
    if scenario.spread_model == "rothermel":
        if scenario.fuel_model_set != "ORIGINAL_13":
            raise ScenarioValidationError(
                f"Rothermel requires fuel_model_set='ORIGINAL_13', but found '{scenario.fuel_model_set}'. "
                "The standard territorial dataset contains experimental Chilean categories, not Anderson 13."
            )

        if scenario.weather.midflame_wind_ms is None:
            raise ScenarioValidationError(
                "Rothermel requires explicit weather.midflame_wind_ms (even if 0.0). "
                "wind_speed_ms cannot be used as midflame without documented conversion."
            )

        fm = scenario.fuel_moisture
        im = scenario.initial_fuel_moisture

        d1 = (getattr(fm, "dead_1h_fraction", None) or getattr(fm, "dead_1h", None)) if fm else None
        d10 = (getattr(fm, "dead_10h_fraction", None) or getattr(fm, "dead_10h", None)) if fm else None
        d100 = (getattr(fm, "dead_100h_fraction", None) or getattr(fm, "dead_100h", None)) if fm else None
        lh = (getattr(fm, "live_herbaceous_fraction", None) or getattr(fm, "live_herbaceous", None)) if fm else None
        lw = (getattr(fm, "live_woody_fraction", None) or getattr(fm, "live_woody", None)) if fm else None

        if im is not None:
            d1 = d1 if d1 is not None else getattr(im, "dead_1h_fraction", None)
            d10 = d10 if d10 is not None else getattr(im, "dead_10h_fraction", None)
            d100 = d100 if d100 is not None else getattr(im, "dead_100h_fraction", None)
            lh = lh if lh is not None else getattr(im, "live_herbaceous_fraction", None)
            lw = lw if lw is not None else getattr(im, "live_woody_fraction", None)

        if (
            d1 is None or
            d10 is None or
            d100 is None or
            lh is None or
            lw is None
        ):
            raise ScenarioValidationError(
                "Rothermel requires an explicit 5-class moisture vector: "
                "dead_1h_fraction, dead_10h_fraction, dead_100h_fraction, "
                "live_herbaceous_fraction, and live_woody_fraction."
            )


        # Validar que todos los códigos de combustible en el ráster pertenezcan a [0..13]
        unique_fuels = np.unique(landscape.fuel_model)
        for fc in unique_fuels:
            if np_isnan(fc):
                continue
            fc_int = int(fc)
            if fc_int < 0 or fc_int > 13:
                raise ScenarioValidationError(
                    f"Ráster contiene código de combustible {fc_int} fuera del rango ORIGINAL_13 (0=sin combustible, 1..13=modelos Anderson)."
                )

        if scenario.rasters.get("propagation_method") == "cumulative":
            raise ScenarioValidationError(
                "Rothermel spread_model only supports Dijkstra minimum arrival time propagation, not 'cumulative'."
            )

    return True, row, col, "Escenario validado correctamente."

def np_isnan(val) -> bool:
    try:
        import numpy as np
        return np.isnan(val)
    except:
        return False

def find_nearest_combustible_cell(landscape: LandscapeData, start_r: int, start_c: int) -> Tuple[int, int]:
    """Busca la celda combustible más cercana utilizando búsqueda en espiral."""
    for radius in range(1, 50):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r, c = start_r + dr, start_c + dc
                if landscape.domain.is_inside(r, c):
                    f_code = landscape.fuel_model[r, c]
                    b_code = landscape.barriers[r, c]
                    if f_code > 0 and b_code not in (1, 5):
                        return r, c
    return 500, 500

def create_corrected_scenario(
    scenario: ScenarioConfig,
    base_dir: Path,
    corrected_row: int,
    corrected_col: int,
    reason: str
):
    """Genera el archivo scenario_001_corrected.json especificando la corrección."""
    output_path = base_dir / "configs" / "scenario_001_corrected.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lon, lat = 0.0, 0.0
    # Obtenemos coords aproximadas para la ignición corregida
    
    scen_dict = scenario.model_dump()
    scen_dict["id"] = "scenario_001_corrected"
    scen_dict["ignition"]["row"] = corrected_row
    scen_dict["ignition"]["col"] = corrected_col
    scen_dict["correction_reason"] = reason
    scen_dict["warning"] = f"Escenario corregido automáticamente por validador: {reason}"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scen_dict, f, indent=2, ensure_ascii=False)

    print(f"[VALIDADOR] Escenario corregido escrito en: {output_path}")
