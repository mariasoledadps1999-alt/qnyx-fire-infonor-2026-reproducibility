"""
Tests for Q-NYX Nelson Physical Model (fire_engine.fuels.nelson).

Includes:
- Unit tests: initialization, trends, invalid inputs, determinism, state containers.
- Golden Reference regression tests: validation against official C++ NFDRS4 fixtures.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List
import pytest

from fire_engine.fuels.nelson import (
    NelsonDFMState,
    NelsonModelConfig,
    NelsonPhysicalModel,
    NFDRS4_REFERENCE_PARAMETERS,
    celsius_to_kelvin,
    cm_to_m,
    cm_to_mm,
    joules_to_cal,
    kelvin_to_celsius,
    m_to_cm,
    mm_to_cm,
    moisture_percent_to_ratio,
    moisture_ratio_to_percent,
    mv_to_w_m2,
    w_m2_to_mv,
)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
BASE_REF_DIR = WORKSPACE_ROOT / "benchmarks" / "nelson_nfdrs4_reference"
WEATHER_DIR = BASE_REF_DIR / "weather"
OFFICIAL_OUT_DIR = BASE_REF_DIR / "official_nfdrs4_output"


# ==============================================================================
# 1. UNIT TESTS: UNIT CONVERSIONS & INITIALIZATION
# ==============================================================================

class TestNelsonUnitConversions:
    def test_length_conversions(self):
        assert cm_to_m(100.0) == 1.0
        assert m_to_cm(1.0) == 100.0
        assert mm_to_cm(10.0) == 1.0
        assert cm_to_mm(1.0) == 10.0

    def test_energy_and_moisture_conversions(self):
        assert pytest.approx(joules_to_cal(4.184), rel=1e-5) == 1.0
        assert pytest.approx(moisture_ratio_to_percent(0.15), rel=1e-5) == 15.0
        assert pytest.approx(moisture_percent_to_ratio(15.0), rel=1e-5) == 0.15

    def test_temperature_conversions(self):
        assert pytest.approx(celsius_to_kelvin(0.0), rel=1e-5) == 273.15
        assert pytest.approx(kelvin_to_celsius(273.15), rel=1e-5) == 0.0

    def test_solar_radiation_conversions(self):
        w_m2 = 94.743
        mv = w_m2_to_mv(w_m2)
        assert pytest.approx(mv, rel=1e-5) == 1.0
        assert pytest.approx(mv_to_w_m2(mv), rel=1e-5) == w_m2


class TestNelsonModelInitialization:
    def test_deterministic_initialization_defaults(self):
        cfg = NelsonModelConfig.for_10h()
        model = NelsonPhysicalModel(cfg)

        state = model.get_state()
        assert state.elapsed_hours == 0.0
        assert state.updates_count == 0
        assert pytest.approx(state.mean_weighted_moisture, abs=1e-6) == 0.20
        assert pytest.approx(state.mean_weighted_moisture_pct, abs=1e-4) == 20.0
        assert pytest.approx(state.surface_moisture, abs=1e-6) == 0.20
        assert pytest.approx(state.surface_temperature_c, abs=1e-6) == 20.0
        assert pytest.approx(state.mean_weighted_temperature_c, abs=1e-6) == 20.0
        assert state.state_id == int(NelsonDFMState.NONE)
        assert len(state.moisture_profile) == 11
        assert len(state.temperature_profile) == 11

    def test_config_factories(self):
        cfg1 = NelsonModelConfig.for_1h()
        assert cfg1.radius_cm == 0.20
        assert cfg1.moisture_substeps == 265

        cfg10 = NelsonModelConfig.for_10h()
        assert cfg10.radius_cm == 0.64
        assert cfg10.moisture_substeps == 60

        cfg100 = NelsonModelConfig.for_100h()
        assert cfg100.radius_cm == 2.00
        assert cfg100.moisture_substeps == 19

    def test_reference_parameters_structure(self):
        for stick in ["1h", "10h", "100h", "1000h"]:
            assert stick in NFDRS4_REFERENCE_PARAMETERS
            p = NFDRS4_REFERENCE_PARAMETERS[stick]
            assert "radius_cm" in p
            assert "moisture_steps" in p
            assert "wmx" in p

    def test_repeated_execution_determinism(self):
        """Two independent instances run with identical weather must produce bitwise identical states."""
        cfg = NelsonModelConfig.for_10h()
        m1 = NelsonPhysicalModel(cfg)
        m2 = NelsonPhysicalModel(cfg)

        wx = {
            "temperature_c": 35.0,
            "relative_humidity_pct": 15.0,
            "solar_radiation_w_m2": 800.0,
            "precipitation_mm": 0.0,
        }
        for _ in range(5):
            m1.update(weather_current=wx, dt_hours=1.0)
            m2.update(weather_current=wx, dt_hours=1.0)

        st1 = m1.get_state()
        st2 = m2.get_state()
        assert st1.mean_weighted_moisture == st2.mean_weighted_moisture
        assert st1.surface_temperature_c == st2.surface_temperature_c
        assert st1.moisture_profile == st2.moisture_profile
        assert st1.temperature_profile == st2.temperature_profile


class TestNelsonPhysicalTrends:
    def test_dry_case_trend(self):
        """Under high T (38°C), low RH (12%), and high solar (900 W/m²), moisture must decline monotonically."""
        model = NelsonPhysicalModel(NelsonModelConfig.for_10h())
        model.initialize(init_moisture=0.20, init_temp_c=20.0)

        wx = {
            "temperature_c": 38.0,
            "relative_humidity_pct": 12.0,
            "solar_radiation_w_m2": 900.0,
            "precipitation_mm": 0.0,
        }

        prev_m = model.get_moisture()
        for step in range(1, 10):
            model.update(weather_current=wx, dt_hours=1.0)
            curr_m = model.get_moisture()
            assert curr_m < prev_m, f"Moisture did not decrease at step {step}: {curr_m} vs {prev_m}"
            prev_m = curr_m

        assert model.get_moisture() < 0.10

    def test_humid_case_trend(self):
        """Under low T (10°C), saturated RH (95%), and 0 solar, dry stick must absorb moisture."""
        model = NelsonPhysicalModel(NelsonModelConfig.for_10h())
        model.initialize(init_moisture=0.08, init_temp_c=10.0, init_rh_fraction=0.95)

        wx = {
            "temperature_c": 10.0,
            "relative_humidity_pct": 95.0,
            "solar_radiation_w_m2": 0.0,
            "precipitation_mm": 0.0,
        }

        prev_m = model.get_moisture()
        for step in range(1, 10):
            model.update(weather_current=wx, dt_hours=1.0)
            curr_m = model.get_moisture()
            assert curr_m > prev_m, f"Moisture did not increase at step {step}: {curr_m} vs {prev_m}"
            prev_m = curr_m

    def test_rain_wetting_increases_surface_moisture(self):
        """Rainfall must rapidly saturate the surface node towards wmx (35%)."""
        model = NelsonPhysicalModel(NelsonModelConfig.for_10h())
        model.initialize(init_moisture=0.10, init_temp_c=15.0)

        wx_rain = {
            "temperature_c": 15.0,
            "relative_humidity_pct": 98.0,
            "solar_radiation_w_m2": 0.0,
            "precipitation_mm": 10.0,  # 10 mm/h heavy rain
        }

        model.update(weather_current=wx_rain, dt_hours=1.0)
        assert model.get_surface_moisture() > 0.25
        assert model.get_state_id() in [int(NelsonDFMState.RAINFALL1), int(NelsonDFMState.RAINFALL2)]


class TestNelsonInvalidInputs:
    def test_invalid_dt_rejected(self):
        model = NelsonPhysicalModel(NelsonModelConfig.for_10h())
        assert model.update(weather_current={"temperature_c": 20.0, "relative_humidity_pct": 50.0}, dt_hours=0.0) is False

    def test_out_of_range_temperature_rejected(self):
        model = NelsonPhysicalModel(NelsonModelConfig.for_10h())
        assert model.update(weather_current={"temperature_c": 85.0, "relative_humidity_pct": 50.0}, dt_hours=1.0) is False
        assert model.update(weather_current={"temperature_c": -75.0, "relative_humidity_pct": 50.0}, dt_hours=1.0) is False

    def test_out_of_range_rh_rejected(self):
        model = NelsonPhysicalModel(NelsonModelConfig.for_10h())
        assert model.update(weather_current={"temperature_c": 20.0, "relative_humidity_pct": 150.0}, dt_hours=1.0) is False


# ==============================================================================
# 2. GOLDEN REFERENCE REGRESSION TESTS (CASES A-E vs OFFICIAL C++ OUTPUT)
# ==============================================================================

def load_official_series(case_id: str, stick_class: str) -> List[float]:
    path = OFFICIAL_OUT_DIR / f"official_{case_id}_{stick_class}.csv"
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [float(r["mean_wtd_moisture_pct"]) for r in reader]


def run_model_on_case(case_id: str, stick_class: str) -> List[float]:
    wx_file = WEATHER_DIR / f"weather_case_{case_id}.csv"
    wx_records = []
    with open(wx_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            wx_records.append({
                "temperature_c": float(r["temperature_c"]),
                "relative_humidity_pct": float(r["relative_humidity_pct"]),
                "solar_radiation_w_m2": float(r["solar_radiation_w_m2"]),
                "precipitation_mm": float(r["precipitation_mm"]),
            })

    cfg = NelsonModelConfig.from_stick_class(stick_class)
    model = NelsonPhysicalModel(cfg)
    model.initialize(init_moisture=0.20, init_temp_c=20.0, init_rh_fraction=0.20)

    results = [model.get_moisture_percent()]
    prev_wx = None
    for curr_wx in wx_records:
        model.update(weather_previous=prev_wx, weather_current=curr_wx, dt_hours=1.0)
        results.append(model.get_moisture_percent())
        prev_wx = curr_wx

    return results


@pytest.mark.skipif(
    not (BASE_REF_DIR / "weather").exists(),
    reason="External C++ NFDRS4 comparison fixtures not present in minimal reproducibility repository"
)
class TestNelsonGoldenReferenceCases:
    """
    Validates Q-NYX NelsonPhysicalModel against the Level 3 Compiled C++ Golden Oracle.
    Tolerances are set based on the quantified cross-validation metrics:
    - 1-h stick: MAE <= 0.01 percentage points (< 1e-4 fraction).
    - 10-h stick: MAE <= 0.08 percentage points.
    - 100-h stick: MAE <= 0.25 percentage points.
    """

    @pytest.mark.parametrize("stick", ["1h", "10h", "100h"])
    def test_golden_case_A_dry_constant(self, stick: str):
        y_qnyx = run_model_on_case("A_dry_constant", stick)
        y_gold = load_official_series("A_dry_constant", stick)
        mae = sum(abs(q - g) for q, g in zip(y_qnyx, y_gold)) / len(y_gold)

        if stick == "1h":
            assert mae <= 0.010, f"Case A 1h MAE too high: {mae:.6f}%"
        elif stick == "10h":
            assert mae <= 0.050, f"Case A 10h MAE too high: {mae:.6f}%"
        else:
            assert mae <= 0.250, f"Case A 100h MAE too high: {mae:.6f}%"

    @pytest.mark.parametrize("stick", ["1h", "10h", "100h"])
    def test_golden_case_B_humid_constant(self, stick: str):
        y_qnyx = run_model_on_case("B_humid_constant", stick)
        y_gold = load_official_series("B_humid_constant", stick)
        mae = sum(abs(q - g) for q, g in zip(y_qnyx, y_gold)) / len(y_gold)

        if stick == "1h":
            assert mae <= 0.005, f"Case B 1h MAE too high: {mae:.6f}%"
        elif stick == "10h":
            assert mae <= 0.010, f"Case B 10h MAE too high: {mae:.6f}%"
        else:
            assert mae <= 0.050, f"Case B 100h MAE too high: {mae:.6f}%"

    @pytest.mark.parametrize("stick", ["1h", "10h", "100h"])
    def test_golden_case_C_diurnal(self, stick: str):
        y_qnyx = run_model_on_case("C_diurnal", stick)
        y_gold = load_official_series("C_diurnal", stick)
        mae = sum(abs(q - g) for q, g in zip(y_qnyx, y_gold)) / len(y_gold)

        if stick == "1h":
            assert mae <= 0.010, f"Case C 1h MAE too high: {mae:.6f}%"
        elif stick == "10h":
            assert mae <= 0.060, f"Case C 10h MAE too high: {mae:.6f}%"
        else:
            assert mae <= 0.120, f"Case C 100h MAE too high: {mae:.6f}%"

    @pytest.mark.parametrize("stick", ["1h", "10h", "100h"])
    def test_golden_case_D_rain_event(self, stick: str):
        y_qnyx = run_model_on_case("D_rain_event", stick)
        y_gold = load_official_series("D_rain_event", stick)
        mae = sum(abs(q - g) for q, g in zip(y_qnyx, y_gold)) / len(y_gold)

        if stick == "1h":
            assert mae <= 0.010, f"Case D 1h MAE too high: {mae:.6f}%"
        elif stick == "10h":
            assert mae <= 0.080, f"Case D 10h MAE too high: {mae:.6f}%"
        else:
            assert mae <= 0.120, f"Case D 100h MAE too high: {mae:.6f}%"

    @pytest.mark.parametrize("stick", ["1h", "10h", "100h"])
    def test_golden_case_E_multiday(self, stick: str):
        y_qnyx = run_model_on_case("E_multiday", stick)
        y_gold = load_official_series("E_multiday", stick)
        mae = sum(abs(q - g) for q, g in zip(y_qnyx, y_gold)) / len(y_gold)

        if stick == "1h":
            assert mae <= 0.010, f"Case E 1h MAE too high: {mae:.6f}%"
        elif stick == "10h":
            assert mae <= 0.060, f"Case E 10h MAE too high: {mae:.6f}%"
        else:
            assert mae <= 0.200, f"Case E 100h MAE too high: {mae:.6f}%"
