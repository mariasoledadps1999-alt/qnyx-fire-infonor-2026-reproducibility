"""
Nelson Dead Fuel Moisture Physical Model and FuelMoistureProvider Adapter.

Implements:
1. The physical model of heat and moisture transfer in cylindrical dead forest fuels
   developed by Ralph M. Nelson Jr. (2000) and operationalized within USFS FireLab
   NFDRS4 (DeadFuelMoisture.cpp).
2. The continuous-time, query-order independent NelsonFuelMoistureProvider adapter
   for simulation engine time series queries FuelMoistureState(t).

Physical references:
- Nelson, R. M. Jr. (2000). Prediction of Diurnal Change in 10-h Fuel Stick
  Moisture Content. Canadian Journal of Forest Research, 30(7): 1071-1087.
- USFS FireLab NFDRS4 DeadFuelMoisture C++ implementation (commit 69d9bab727).
- USDA Forest Service General Technical Report RMRS-GTR-379 (2018).

NOTE:
This module is fully isolated and decoupled from SpreadModel and Rothermel.
Default parameters represent NFDRS4 reference baseline values for model
verification against the official C++ golden oracle; they do NOT yet represent
calibrated parameters for Chilean fuels (e.g., Pinus radiata, Eucalyptus).
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from fire_engine.fuels.moisture import (
    FuelMoistureError,
    FuelMoistureProvider,
    FuelMoistureState,
    InvalidFuelMoistureValueError,
)
from fire_engine.weather import MissingWeatherRecordError, WeatherRangeError


# ==============================================================================
# EXCEPTIONS
# ==============================================================================

class NelsonError(FuelMoistureError):
    """Base exception for Nelson fuel moisture physical model errors."""
    pass


class NelsonRequiredWeatherInputMissingError(NelsonError):
    """Raised when an atmospheric input required by Nelson physics (e.g. solar radiation) is missing."""
    pass


# ==============================================================================
# EXPLICIT UNIT CONVERSION FUNCTIONS
# ==============================================================================

def cm_to_m(cm: float) -> float:
    """Convert centimeters to meters."""
    return float(cm) / 100.0


def m_to_cm(m: float) -> float:
    """Convert meters to centimeters."""
    return float(m) * 100.0


def mm_to_cm(mm: float) -> float:
    """Convert millimeters to centimeters."""
    return float(mm) / 10.0


def cm_to_mm(cm: float) -> float:
    """Convert centimeters to millimeters."""
    return float(cm) * 10.0


def cal_to_joules(cal: float) -> float:
    """Convert thermochemical calories to Joules (1 cal = 4.184 J)."""
    return float(cal) * 4.184


def joules_to_cal(joules: float) -> float:
    """Convert Joules to thermochemical calories."""
    return float(joules) / 4.184


def moisture_ratio_to_percent(ratio: float) -> float:
    """Convert dimensionless moisture fraction (g/g) to percentage (%)."""
    return float(ratio) * 100.0


def moisture_percent_to_ratio(pct: float) -> float:
    """Convert moisture percentage (%) to dimensionless fraction (g/g)."""
    return float(pct) / 100.0


def celsius_to_kelvin(celsius: float) -> float:
    """Convert temperature from Celsius to Kelvin."""
    return float(celsius) + 273.15


def kelvin_to_celsius(kelvin: float) -> float:
    """Convert temperature from Kelvin to Celsius."""
    return float(kelvin) - 273.15


def w_m2_to_mv(w_m2: float) -> float:
    """
    Convert solar radiation flux in W/m^2 to pyranometer millivolts (mV)
    using the NFDRS RAWS pyranometer calibration constant (Smv = 94.743 W/m^2/mV).
    """
    return float(w_m2) / 94.743


def mv_to_w_m2(mv: float) -> float:
    """Convert pyranometer millivolts (mV) to solar radiation flux in W/m^2."""
    return float(mv) * 94.743


# ==============================================================================
# ENUMERATIONS & STATE DEFINITIONS
# ==============================================================================

class NelsonDFMState(IntEnum):
    """Surface exchange state identifiers corresponding to NFDRS4 DFM states."""
    NONE = 0
    ADSORPTION = 1
    DESORPTION = 2
    CONDENSATION1 = 3
    CONDENSATION2 = 4
    EVAPORATION = 5
    RAINFALL1 = 6
    RAINFALL2 = 7
    RAINSTORM = 8
    STAGNATION = 9
    ERROR = 10


NELSON_STATE_NAMES = [
    "None",
    "Adsorption",
    "Desorption",
    "Condensation1",
    "Condensation2",
    "Evaporation",
    "Rainfall1",
    "Rainfall2",
    "Rainstorm",
    "Stagnation",
    "Error",
]


# ==============================================================================
# NFDRS4 REFERENCE PARAMETERS (USFS FIRELAB STANDARD BASELINE)
# ==============================================================================

NFDRS4_REFERENCE_PARAMETERS: Dict[str, Dict[str, Any]] = {
    "1h": {
        "description": "NFDRS4 1-hour dead fuel stick (d=4 mm, r=0.20 cm)",
        "radius_cm": 0.20,
        "stick_nodes": 11,
        "moisture_steps": 265,
        "diffusivity_steps": 24,
        "adsorption_rate": 0.462252733,
        "desorption_rate": 0.060000000,
        "planar_heat_transfer_rate": 3.1598,
        "runoff_factor": 3.6873,
        "density_g_cm3": 0.40,
        "length_cm": 41.0,
        "wmx": 0.35,
    },
    "10h": {
        "description": "NFDRS4 10-hour dead fuel stick (d=12.8 mm, r=0.64 cm)",
        "radius_cm": 0.64,
        "stick_nodes": 11,
        "moisture_steps": 60,
        "diffusivity_steps": 9,
        "adsorption_rate": 0.079548303,
        "desorption_rate": 0.060000000,
        "planar_heat_transfer_rate": 0.3797,
        "runoff_factor": 0.3106,
        "density_g_cm3": 0.40,
        "length_cm": 41.0,
        "wmx": 0.35,
    },
    "100h": {
        "description": "NFDRS4 100-hour dead fuel stick (d=40 mm, r=2.00 cm)",
        "radius_cm": 2.00,
        "stick_nodes": 11,
        "moisture_steps": 19,
        "diffusivity_steps": 5,
        "adsorption_rate": 0.060000000,
        "desorption_rate": 0.060000000,
        "planar_heat_transfer_rate": 0.2288,
        "runoff_factor": 0.0512,
        "density_g_cm3": 0.40,
        "length_cm": 41.0,
        "wmx": 0.35,
    },
    "1000h": {
        "description": "NFDRS4 1000-hour dead fuel stick (d=76.2 mm, r=3.81 cm)",
        "radius_cm": 3.81,
        "stick_nodes": 11,
        "moisture_steps": 14,
        "diffusivity_steps": 5,
        "adsorption_rate": 0.060000000,
        "desorption_rate": 0.060000000,
        "planar_heat_transfer_rate": 0.2206,
        "runoff_factor": 0.0339,
        "density_g_cm3": 0.40,
        "length_cm": 41.0,
        "wmx": 0.35,
    },
}


# ==============================================================================
# CONFIGURATION & STATE CONTAINERS
# ==============================================================================

@dataclass
class NelsonModelConfig:
    """
    Configuration parameters for the Nelson Dead Fuel Moisture Model.
    """
    stick_class: str = "10h"
    radius_cm: Optional[float] = None
    density_g_cm3: float = 0.40
    length_cm: float = 41.0
    wmx: float = 0.35
    nodes: Optional[int] = None
    moisture_substeps: Optional[int] = None
    diffusivity_substeps: Optional[int] = None
    solver_mode: str = "nfdrs_compatible"
    allow_rainfall2: bool = False
    allow_rainstorm: bool = False
    ramp_rai0: bool = False

    def __post_init__(self):
        sc = self.stick_class.lower().strip()
        if sc in NFDRS4_REFERENCE_PARAMETERS:
            ref = NFDRS4_REFERENCE_PARAMETERS[sc]
            if self.radius_cm is None:
                self.radius_cm = ref["radius_cm"]
            if self.nodes is None:
                self.nodes = ref["stick_nodes"]
            if self.moisture_substeps is None:
                self.moisture_substeps = ref["moisture_steps"]
            if self.diffusivity_substeps is None:
                self.diffusivity_substeps = ref["diffusivity_steps"]
        else:
            if self.radius_cm is None:
                raise ValueError(f"Unknown stick_class '{self.stick_class}'; explicit radius_cm is required.")
            if self.nodes is None:
                self.nodes = derive_stick_nodes(self.radius_cm)
            if self.moisture_substeps is None:
                self.moisture_substeps = derive_moisture_steps(self.radius_cm)
            if self.diffusivity_substeps is None:
                self.diffusivity_substeps = derive_diffusivity_steps(self.radius_cm)

    @classmethod
    def for_1h(cls, **kwargs) -> NelsonModelConfig:
        return cls(stick_class="1h", **kwargs)

    @classmethod
    def for_10h(cls, **kwargs) -> NelsonModelConfig:
        return cls(stick_class="10h", **kwargs)

    @classmethod
    def for_100h(cls, **kwargs) -> NelsonModelConfig:
        return cls(stick_class="100h", **kwargs)

    @classmethod
    def for_1000h(cls, **kwargs) -> NelsonModelConfig:
        return cls(stick_class="1000h", **kwargs)

    @classmethod
    def from_stick_class(cls, stick_class: str, **kwargs) -> NelsonModelConfig:
        return cls(stick_class=stick_class, **kwargs)


@dataclass
class NelsonFuelStickState:
    """
    Represents the full internal and surface state of a cylindrical fuel stick.
    """
    elapsed_hours: float
    updates_count: int
    mean_weighted_moisture: float
    mean_weighted_moisture_pct: float
    median_radial_moisture: float
    median_radial_moisture_pct: float
    surface_moisture: float
    surface_moisture_pct: float
    surface_temperature_c: float
    mean_weighted_temperature_c: float
    surface_film: float
    state_id: int
    state_name: str
    moisture_profile: Tuple[float, ...] = field(default_factory=tuple)
    temperature_profile: Tuple[float, ...] = field(default_factory=tuple)
    diffusivity_profile: Tuple[float, ...] = field(default_factory=tuple)
    saturation_profile: Tuple[float, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class NelsonCheckpointState:
    """
    Immutable, deep-copy safe snapshot of a single cylindrical fuel stick state.
    Contains everything needed to deterministically restore and advance the stick.
    """
    stick_class: str
    radius_cm: float
    density_g_cm3: float
    length_cm: float
    nodes: int
    moisture_substeps: int
    diffusivity_substeps: int
    wmx: float
    # Radial profiles
    moisture_profile: Tuple[float, ...]
    temperature_profile: Tuple[float, ...]
    saturation_profile: Tuple[float, ...]
    diffusivity_profile: Tuple[float, ...]
    # Atmospheric memory from previous step
    ta0: float
    ta1: float
    ha0: float
    ha1: float
    sv0: float
    sv1: float
    rc0: float
    rc1: float
    ra0: float
    ra1: float
    bp0: float
    bp1: float
    # Surface state variables
    hf: float
    wsa: float
    sem: float
    wfilm: float
    rdur: float
    elapsed: float
    updates: int
    state_id: int


@dataclass(frozen=True)
class NelsonConditioningReport:
    """
    Summary report of the Nelson physical conditioning period [-N, 0).
    """
    conditioning_hours: int
    weather_records_used: int
    seed_mode: str
    seed_dead_1h: float
    seed_dead_10h: float
    seed_dead_100h: float
    final_dead_1h: float
    final_dead_10h: float
    final_dead_100h: float
    final_temperature_1h_c: float
    final_temperature_10h_c: float
    final_temperature_100h_c: float
    weather_data_start_datetime_utc: Optional[str] = None
    simulation_start_datetime_utc: Optional[str] = None
    conditioning_start_datetime_utc: Optional[str] = None


@dataclass(frozen=True)
class NelsonTriStickCheckpoint:
    """
    Frozen hourly boundary checkpoint holding the exact states of 1h, 10h, and 100h sticks.
    """
    elapsed_minutes: float
    hour_index: int
    chk_1h: NelsonCheckpointState
    chk_10h: NelsonCheckpointState
    chk_100h: NelsonCheckpointState
    fuel_moisture_state: FuelMoistureState
    weather_record: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class NelsonFuelMoistureTimeSeries:
    """
    Precomputed contiguous runtime lookup table for dead and live fuel moisture.
    Separates runtime numerical lookup from the physical radial checkpoint state.

    Attributes:
        times_minutes: Contiguous 1D float64 array of sample timestamps [0, dt, 2dt, ..., total_duration_minutes].
        dead_1h: Contiguous 1D float64 array of 1-hour dead fuel moisture (dimensionless ratio g/g).
        dead_10h: Contiguous 1D float64 array of 10-hour dead fuel moisture (dimensionless ratio g/g).
        dead_100h: Contiguous 1D float64 array of 100-hour dead fuel moisture (dimensionless ratio g/g).
        live_herbaceous: Constant live herbaceous moisture ratio (g/g).
        live_woody: Constant live woody moisture ratio (g/g).
        dt_minutes: Nominal step resolution between samples (minutes).
        total_duration_minutes: Total simulation duration covered (minutes).
        interpolation_method: "linear" or "previous" (step).
    """
    times_minutes: np.ndarray
    dead_1h: np.ndarray
    dead_10h: np.ndarray
    dead_100h: np.ndarray
    live_herbaceous: float
    live_woody: float
    dt_minutes: float
    total_duration_minutes: float
    interpolation_method: str = "linear"
    is_uniform: bool = True

    def __post_init__(self):
        # Validate and enforce contiguous read-only arrays
        for arr_name in ("times_minutes", "dead_1h", "dead_10h", "dead_100h"):
            arr = getattr(self, arr_name)
            if not isinstance(arr, np.ndarray):
                arr = np.ascontiguousarray(arr, dtype=np.float64)
                object.__setattr__(self, arr_name, arr)
            arr.flags.writeable = False

        method = self.interpolation_method.lower().strip()
        if method not in ("linear", "previous", "step"):
            raise ValueError(
                f"Unsupported interpolation_method '{self.interpolation_method}'. Use 'linear' or 'previous'."
            )
        object.__setattr__(self, "interpolation_method", method)

        # Check grid uniformity
        if len(self.times_minutes) > 2 and self.dt_minutes > 0.0:
            diffs = np.diff(self.times_minutes)
            uniform_check = bool(np.allclose(diffs, self.dt_minutes, atol=1e-5))
            object.__setattr__(self, "is_uniform", uniform_check)
        else:
            object.__setattr__(self, "is_uniform", True)

        if not self.is_uniform:
            object.__setattr__(self, "_times_tuple", tuple(self.times_minutes))
        else:
            object.__setattr__(self, "_times_tuple", None)

    def memory_bytes(self) -> Dict[str, Any]:
        """Returns the memory footprint in bytes, separating raw ndarray buffers from Python overhead."""
        raw_bytes = int(
            self.times_minutes.nbytes +
            self.dead_1h.nbytes +
            self.dead_10h.nbytes +
            self.dead_100h.nbytes
        )
        return {
            "num_points": len(self.times_minutes),
            "raw_ndarray_bytes": raw_bytes,
            "total_bytes_approx": raw_bytes + 512,  # Python object wrapper overhead
            "dt_minutes": self.dt_minutes,
            "total_duration_minutes": self.total_duration_minutes,
            "is_uniform": self.is_uniform,
        }

    def get_state(
        self,
        elapsed_minutes: float = 0.0,
        row: Optional[int] = None,
        col: Optional[int] = None,
    ) -> FuelMoistureState:
        """Alias for get_state_fast to satisfy the FuelMoistureProvider interface."""
        return self.get_state_fast(elapsed_minutes, row, col)

    def get_state_fast(
        self,
        elapsed_minutes: float = 0.0,
        row: Optional[int] = None,
        col: Optional[int] = None,
    ) -> FuelMoistureState:
        """
        Ultra-fast runtime lookup of FuelMoistureState.
        Performs direct index calculation without executing ODE integration:
        - Uniform grids: O(1) arithmetic indexing (<0.4 us).
        - Adaptive non-uniform grids: O(log N) binary search on immutable tuple (<0.7 us).
        """
        t = float(elapsed_minutes)
        if t < 0.0:
            raise WeatherRangeError(f"Negative simulation time not allowed: {t} min.")

        max_t = self.total_duration_minutes
        if t > max_t:
            if t - max_t < 1e-9:
                t = max_t
            else:
                raise WeatherRangeError(
                    f"Requested simulation time {t:.2f} min exceeds available range ({max_t:.1f} min)."
                )

        n_points = len(self.times_minutes)
        if n_points == 1 or math.isclose(t, 0.0, abs_tol=1e-12):
            return FuelMoistureState(
                dead_1h=float(self.dead_1h[0]),
                dead_10h=float(self.dead_10h[0]),
                dead_100h=float(self.dead_100h[0]),
                live_herbaceous=self.live_herbaceous,
                live_woody=self.live_woody,
            )

        if math.isclose(t, max_t, abs_tol=1e-12):
            return FuelMoistureState(
                dead_1h=float(self.dead_1h[-1]),
                dead_10h=float(self.dead_10h[-1]),
                dead_100h=float(self.dead_100h[-1]),
                live_herbaceous=self.live_herbaceous,
                live_woody=self.live_woody,
            )

        if self.is_uniform:
            # O(1) index computation based on nominal dt
            idx = int(t / self.dt_minutes)
            if idx >= n_points - 1:
                idx = n_points - 2

            t0 = self.times_minutes[idx]
            t1 = self.times_minutes[idx + 1]

            # Guard against minor index boundary shifts if non-uniform last step
            if t < t0 and idx > 0:
                idx -= 1
                t0 = self.times_minutes[idx]
                t1 = self.times_minutes[idx + 1]
            elif t > t1 and idx < n_points - 2:
                idx += 1
                t0 = self.times_minutes[idx]
                t1 = self.times_minutes[idx + 1]
        else:
            # Non-uniform binary search using bisect on tuple (0.6 us)
            idx = bisect.bisect_right(self._times_tuple, t) - 1
            if idx < 0:
                idx = 0
            elif idx >= n_points - 1:
                idx = n_points - 2
            t0 = self._times_tuple[idx]
            t1 = self._times_tuple[idx + 1]

        if self.interpolation_method in ("previous", "step") or math.isclose(t, t0, abs_tol=1e-12):
            m1 = self.dead_1h[idx]
            m10 = self.dead_10h[idx]
            m100 = self.dead_100h[idx]
        elif math.isclose(t, t1, abs_tol=1e-12):
            m1 = self.dead_1h[idx + 1]
            m10 = self.dead_10h[idx + 1]
            m100 = self.dead_100h[idx + 1]
        else:
            delta = t1 - t0
            u = (t - t0) / delta if delta > 1e-12 else 0.0
            m1 = self.dead_1h[idx] + u * (self.dead_1h[idx + 1] - self.dead_1h[idx])
            m10 = self.dead_10h[idx] + u * (self.dead_10h[idx + 1] - self.dead_10h[idx])
            m100 = self.dead_100h[idx] + u * (self.dead_100h[idx + 1] - self.dead_100h[idx])

        return FuelMoistureState(
            dead_1h=float(m1),
            dead_10h=float(m10),
            dead_100h=float(m100),
            live_herbaceous=self.live_herbaceous,
            live_woody=self.live_woody,
        )



# ==============================================================================
# DERIVATION FUNCTIONS FOR RADIAL DISCRETIZATION & PARAMETERS
# ==============================================================================

def derive_stick_nodes(radius_cm: float) -> int:
    """Derive odd number of radial nodes based on stick radius (cm)."""
    nodes = int(10.727 + 0.1746 / float(radius_cm))
    if nodes % 2 == 0:
        nodes += 1
    return nodes


def derive_moisture_steps(radius_cm: float) -> int:
    """Derive hourly moisture sub-steps based on stick radius (cm)."""
    return int(9.8202 + 26.865 / math.pow(float(radius_cm), 1.4))


def derive_diffusivity_steps(radius_cm: float) -> int:
    """Derive hourly diffusivity sub-steps based on stick radius (cm)."""
    return int(4.777 + 2.496 / math.pow(float(radius_cm), 1.3))


def derive_planar_heat_transfer_rate(radius_cm: float) -> float:
    """Derive planar convective heat transfer rate hc (cal/cm^2-h-C)."""
    return 0.2195 + 0.05260 / math.pow(float(radius_cm), 2.5)


def derive_adsorption_rate(radius_cm: float) -> float:
    """Derive boundary mass transfer rate for adsorption alpha (cm/h)."""
    return 0.06 + 0.006126 / math.pow(float(radius_cm), 2.6)


def derive_rainfall_runoff_factor(radius_cm: float) -> float:
    """Derive rainfall surface retention factor f0."""
    return 0.02822 + 0.1056 / math.pow(float(radius_cm), 2.2)


# ==============================================================================
# NELSON PHYSICAL MODEL
# ==============================================================================

class NelsonPhysicalModel:
    """
    Nelson (2000) 1-D Radial Dead Fuel Moisture & Heat Transfer Model.

    Simulates heat and moisture diffusion across concentric cylindrical shells
    with dynamic surface boundary conditions driven by air temperature, relative
    humidity, solar radiation, and precipitation.
    """

    # Thermodynamic constants (NFDRS4 C++ DeadFuelMoisture)
    Aks: float = 2.0e-13
    Alb: float = 0.6
    Alpha: float = 0.25
    Ap: float = 0.000772
    Aw: float = 0.8
    Eps: float = 0.85
    Hfs: float = 0.99
    Kelvin: float = 273.2
    Pi: float = 3.141592654
    Pr: float = 0.7
    Sbc: float = 1.37e-12
    Sc: float = 0.58
    Smv: float = 94.743
    St: float = 72.8
    Tcd: float = 6.0
    Tcn: float = 3.0
    Thdiff: float = 8.0
    Wl: float = 0.0023
    Srf: float = 14.82052
    Wsf: float = 4.60517
    Hrd: float = 0.116171
    Hrn: float = 0.112467
    Sir: float = 0.0714285
    Scr: float = 0.285714

    def __init__(self, config: Optional[NelsonModelConfig] = None):
        self.config: NelsonModelConfig = config or NelsonModelConfig.for_10h()
        self.m_radius: float = self.config.radius_cm
        self.m_density: float = self.config.density_g_cm3
        self.m_length: float = self.config.length_cm
        self.m_nodes: int = self.config.nodes
        self.m_mSteps: int = self.config.moisture_substeps
        self.m_dSteps: int = self.config.diffusivity_substeps
        self.m_wmx: float = self.config.wmx

        # Transfer rates & derived factors
        self.m_hc: float = derive_planar_heat_transfer_rate(self.m_radius)
        self.m_stca: float = derive_adsorption_rate(self.m_radius)
        self.m_stcd: float = 0.06  # desorption rate is constant 0.06 cm/h
        self.m_rai0: float = derive_rainfall_runoff_factor(self.m_radius)
        self.m_rai1: float = 0.5
        self.m_stv: float = 9999.0
        self.m_wfilmk: float = 0.0
        self.m_allowRainfall2: bool = self.config.allow_rainfall2
        self.m_allowRainstorm: bool = self.config.allow_rainstorm
        self.m_rampRai0: bool = self.config.ramp_rai0

        # Special overrides from NFDRS4 standard parameter tables
        sc = self.config.stick_class.lower().strip()
        if sc in NFDRS4_REFERENCE_PARAMETERS:
            ref = NFDRS4_REFERENCE_PARAMETERS[sc]
            self.m_stca = ref["adsorption_rate"]
            self.m_stcd = ref["desorption_rate"]

        # Grid discretization
        self.m_dx: float = self.m_radius / float(self.m_nodes - 1)
        self.m_dx_2: float = self.m_dx * 2.0
        self.m_wmax: float = (1.0 / self.m_density) - (1.0 / 1.53)

        self.m_x: List[float] = []
        for i in range(self.m_nodes - 1):
            self.m_x.append(self.m_radius - (self.m_dx * i))
        self.m_x.append(0.0)

        # Concentric shell volume fractions (sum = 1.0)
        self.m_v: List[float] = []
        ro = self.m_radius
        ri = ro - 0.5 * self.m_dx
        a2 = self.m_radius * self.m_radius
        self.m_v.append((ro * ro - ri * ri) / a2)
        for _ in range(1, self.m_nodes - 1):
            ro = ri
            ri = ro - self.m_dx
            self.m_v.append((ro * ro - ri * ri) / a2)
        self.m_v.append((ri * ri) / a2)

        # Precalculated convective and psychrometric constants
        self.m_hwf: float = 0.622 * self.m_hc * math.pow((self.Pr / self.Sc), 0.667)
        self.m_amlf: float = self.m_hwf / (0.24 * self.m_density * self.m_radius)
        rcav = 0.5 * self.Aw * self.Wl
        self.m_capf: float = (
            3600.0
            * self.Pi
            * self.St
            * rcav
            * rcav
            / (16.0 * self.m_radius * self.m_radius * self.m_length * self.m_density)
        )
        self.m_vf: float = self.St / (self.m_density * self.Wl * self.Scr)

        # Working state arrays
        self.m_t: List[float] = [20.0] * self.m_nodes
        self.m_w: List[float] = [0.20] * self.m_nodes
        self.m_s: List[float] = [0.0] * self.m_nodes
        self.m_d: List[float] = [0.0] * self.m_nodes

        self.m_Twold: List[float] = [0.0] * self.m_nodes
        self.m_Tsold: List[float] = [0.0] * self.m_nodes
        self.m_Ttold: List[float] = [0.0] * self.m_nodes
        self.m_Tv: List[float] = [0.0] * self.m_nodes
        self.m_To: List[float] = [0.0] * self.m_nodes
        self.m_Tg: List[float] = [0.0] * self.m_nodes

        # Atmospheric history states
        self.m_ta0: float = 20.0
        self.m_ta1: float = 20.0
        self.m_ha0: float = 0.20
        self.m_ha1: float = 0.20
        self.m_sv0: float = 0.0
        self.m_sv1: float = 0.0
        self.m_rc0: float = 0.0
        self.m_rc1: float = 0.0
        self.m_ra0: float = 0.0
        self.m_ra1: float = 0.0
        self.m_bp0: float = 0.0218
        self.m_bp1: float = 0.0218

        self.m_hf: float = 0.20
        self.m_wsa: float = 0.30
        self.m_sem: float = 0.05
        self.m_wfilm: float = 0.0
        self.m_rdur: float = 0.0
        self.m_elapsed: float = 0.0
        self.m_updates: int = 0
        self.m_state: int = int(NelsonDFMState.NONE)

        # Default initialization
        self.initialize(init_moisture=0.20, init_temp_c=20.0, init_rh_fraction=0.20)

    @classmethod
    def for_1h(cls, **kwargs) -> NelsonPhysicalModel:
        return cls(NelsonModelConfig.for_1h(**kwargs))

    @classmethod
    def for_10h(cls, **kwargs) -> NelsonPhysicalModel:
        return cls(NelsonModelConfig.for_10h(**kwargs))

    @classmethod
    def for_100h(cls, **kwargs) -> NelsonPhysicalModel:
        return cls(NelsonModelConfig.for_100h(**kwargs))

    @classmethod
    def for_1000h(cls, **kwargs) -> NelsonPhysicalModel:
        return cls(NelsonModelConfig.for_1000h(**kwargs))

    def initialize(
        self,
        init_moisture: float = 0.20,
        init_temp_c: float = 20.0,
        init_rh_fraction: float = 0.20,
        barometric_pressure: float = 0.0218,
    ) -> None:
        """
        Initializes or resets the fuel stick state uniformly across all radial nodes.
        """
        w_val = max(0.001, min(float(init_moisture), self.m_wmx))
        t_val = float(init_temp_c)
        ha_val = max(0.001, min(float(init_rh_fraction), 1.0))
        bp_val = float(barometric_pressure)

        self.m_ta0 = self.m_ta1 = t_val
        self.m_ha0 = self.m_ha1 = ha_val
        self.m_sv0 = self.m_sv1 = 0.0
        self.m_rc0 = self.m_rc1 = 0.0
        self.m_ra0 = self.m_ra1 = 0.0
        self.m_bp0 = self.m_bp1 = bp_val

        self.m_hf = min(ha_val, self.Hfs)
        self.m_wfilm = 0.0
        self.m_wsa = w_val + 0.1
        self.m_t = [t_val] * self.m_nodes
        self.m_w = [w_val] * self.m_nodes
        self.m_s = [0.0] * self.m_nodes

        self.m_rdur = 0.0
        self.m_elapsed = 0.0
        self.m_updates = 0
        self.m_state = int(NelsonDFMState.NONE)

        # Initial radial diffusivity distribution
        self._calculate_diffusivity(self.m_bp0)

    def reset(self, init_moisture: float = 0.20) -> None:
        """Reset state back to initial moisture condition."""
        self.initialize(init_moisture=init_moisture)

    # --------------------------------------------------------------------------
    # Checkpoint & Cloning Architecture (NELSON-002B)
    # --------------------------------------------------------------------------
    def get_checkpoint(self) -> NelsonCheckpointState:
        """
        Creates an immutable, deep-copy safe snapshot of the fuel stick state.
        """
        return NelsonCheckpointState(
            stick_class=self.config.stick_class,
            radius_cm=self.m_radius,
            density_g_cm3=self.m_density,
            length_cm=self.m_length,
            nodes=self.m_nodes,
            moisture_substeps=self.m_mSteps,
            diffusivity_substeps=self.m_dSteps,
            wmx=self.m_wmx,
            moisture_profile=tuple(self.m_w),
            temperature_profile=tuple(self.m_t),
            saturation_profile=tuple(self.m_s),
            diffusivity_profile=tuple(self.m_d),
            ta0=self.m_ta0,
            ta1=self.m_ta1,
            ha0=self.m_ha0,
            ha1=self.m_ha1,
            sv0=self.m_sv0,
            sv1=self.m_sv1,
            rc0=self.m_rc0,
            rc1=self.m_rc1,
            ra0=self.m_ra0,
            ra1=self.m_ra1,
            bp0=self.m_bp0,
            bp1=self.m_bp1,
            hf=self.m_hf,
            wsa=self.m_wsa,
            sem=self.m_sem,
            wfilm=self.m_wfilm,
            rdur=self.m_rdur,
            elapsed=self.m_elapsed,
            updates=self.m_updates,
            state_id=self.m_state,
        )

    def restore_checkpoint(self, chk: NelsonCheckpointState) -> None:
        """
        Restores the exact physical state from a checkpoint.
        """
        self.m_radius = chk.radius_cm
        self.m_density = chk.density_g_cm3
        self.m_length = chk.length_cm
        self.m_nodes = chk.nodes
        self.m_mSteps = chk.moisture_substeps
        self.m_dSteps = chk.diffusivity_substeps
        self.m_wmx = chk.wmx
        self.m_w = list(chk.moisture_profile)
        self.m_t = list(chk.temperature_profile)
        self.m_s = list(chk.saturation_profile)
        self.m_d = list(chk.diffusivity_profile)
        self.m_ta0 = chk.ta0
        self.m_ta1 = chk.ta1
        self.m_ha0 = chk.ha0
        self.m_ha1 = chk.ha1
        self.m_sv0 = chk.sv0
        self.m_sv1 = chk.sv1
        self.m_rc0 = chk.rc0
        self.m_rc1 = chk.rc1
        self.m_ra0 = chk.ra0
        self.m_ra1 = chk.ra1
        self.m_bp0 = chk.bp0
        self.m_bp1 = chk.bp1
        self.m_hf = chk.hf
        self.m_wsa = chk.wsa
        self.m_sem = chk.sem
        self.m_wfilm = chk.wfilm
        self.m_rdur = chk.rdur
        self.m_elapsed = chk.elapsed
        self.m_updates = chk.updates
        self.m_state = chk.state_id

    @classmethod
    def from_checkpoint(cls, chk: NelsonCheckpointState) -> NelsonPhysicalModel:
        """
        Constructs a new independent NelsonPhysicalModel restored from a checkpoint.
        """
        cfg = NelsonModelConfig(
            stick_class=chk.stick_class,
            radius_cm=chk.radius_cm,
            density_g_cm3=chk.density_g_cm3,
            length_cm=chk.length_cm,
            wmx=chk.wmx,
            nodes=chk.nodes,
            moisture_substeps=chk.moisture_substeps,
            diffusivity_substeps=chk.diffusivity_substeps,
        )
        model = cls(cfg)
        model.restore_checkpoint(chk)
        return model

    def clone(self) -> NelsonPhysicalModel:
        """
        Returns a completely independent copy of this model instance.
        """
        return NelsonPhysicalModel.from_checkpoint(self.get_checkpoint())

    # --------------------------------------------------------------------------
    # Internal Physics: Diffusivity Calculation
    # --------------------------------------------------------------------------
    def _calculate_diffusivity(self, bp: float) -> None:
        """
        Calculates bound water and water vapor diffusion coefficients
        at each radial node following Nelson (2000).
        """
        for i in range(self.m_nodes):
            tk = self.m_t[i] + 273.2
            qv = 13550.0 - 10.22 * tk
            cpv = 7.22 + 0.002374 * tk + 2.67e-07 * tk * tk
            dv = 0.22 * 3600.0 * (0.0242 / bp) * math.pow((tk / 273.2), 1.75)
            ps1 = 0.0000239 * math.exp(20.58 - (5205.0 / tk))
            c1 = 0.1617 - 0.001419 * self.m_t[i]
            c2 = 0.4657 + 0.003578 * self.m_t[i]

            dhdm = 0.0
            if self.m_w[i] < self.m_wsa:
                wc = self.m_w[i]
                if c2 != 1.0 and self.m_hf < 1.0 and c1 != 0.0 and c2 != 0.0:
                    dhdm = (
                        (1.0 - self.m_hf)
                        * math.pow(-math.log(1.0 - self.m_hf), (1.0 - c2))
                        / (c1 * c2)
                    )
            else:
                wc = self.m_wsa
                if c2 != 1.0 and self.Hfs < 1.0 and c1 != 0.0 and c2 != 0.0:
                    dhdm = (
                        (1.0 - self.Hfs)
                        * math.pow(self.Wsf, (1.0 - c2))
                        / (c1 * c2)
                    )

            daw = 1.3 - 0.64 * wc
            svaw = 1.0 / daw
            vfaw = svaw * wc / (0.685 + svaw * wc)
            vfcw = (0.685 + svaw * wc) / ((1.0 / self.m_density) + svaw * wc)
            rfcw = 1.0 - math.sqrt(max(0.0, 1.0 - vfcw))
            fac = 1.0 / max(1e-12, rfcw * vfcw)
            con = 1.0 / (2.0 - vfaw)
            qw = 5040.0 * math.exp(-14.0 * wc)
            e = (qv + qw - cpv * tk) / 1.2

            dvpr = (
                18.0
                * 0.016
                * (1.0 - vfcw)
                * dv
                * ps1
                * dhdm
                / (self.m_density * 1.987 * tk)
            )
            self.m_d[i] = dvpr + 3600.0 * 0.0985 * con * fac * math.exp(
                -e / (1.987 * tk)
            )

    # --------------------------------------------------------------------------
    # Dynamic Update Method
    # --------------------------------------------------------------------------
    def update(
        self,
        weather_previous: Optional[Union[Dict[str, Any], Any]] = None,
        weather_current: Optional[Union[Dict[str, Any], Any]] = None,
        dt_hours: float = 1.0,
        precipitation_amount_cm: Optional[float] = None,
        barometric_pressure: float = 0.0218,
    ) -> bool:
        """
        Advances the physical fuel moisture and temperature state across interval dt_hours.

        Accepts either:
        - Weather objects/dictionaries (with keys/attributes 'temperature_c',
          'relative_humidity_pct', 'solar_radiation_w_m2', 'precipitation_mm')
        - Or scalar atmospheric values directly through keyword arguments.
        """
        et = float(dt_hours)
        if et < 0.0000027:
            return False

        # Extract previous observation
        if weather_previous is not None:
            ta0 = self._extract_attr(weather_previous, "temperature_c", default=self.m_ta1)
            rh0 = self._extract_attr(weather_previous, "relative_humidity_pct", default=self.m_ha1 * 100.0)
            sr0 = self._extract_attr(weather_previous, "solar_radiation_w_m2", default=self.m_sv1 * self.Smv)
            self.m_ta0 = float(ta0)
            self.m_ha0 = float(rh0) / 100.0 if float(rh0) > 1.0 else float(rh0)
            self.m_sv0 = float(sr0) / self.Smv
        else:
            self.m_ta0 = self.m_ta1
            self.m_ha0 = self.m_ha1
            self.m_sv0 = self.m_sv1

        # Extract current observation
        if weather_current is not None:
            ta1 = self._extract_attr(weather_current, "temperature_c", default=self.m_ta0)
            rh1 = self._extract_attr(weather_current, "relative_humidity_pct", default=self.m_ha0 * 100.0)
            sr1 = self._extract_attr(weather_current, "solar_radiation_w_m2", default=self.m_sv0 * self.Smv)
            precip_mm = self._extract_attr(weather_current, "precipitation_mm", default=0.0)
            rcum_cm = mm_to_cm(precip_mm) if precipitation_amount_cm is None else float(precipitation_amount_cm)
        else:
            ta1 = self.m_ta0
            rh1 = self.m_ha0 * 100.0
            sr1 = self.m_sv0 * self.Smv
            rcum_cm = 0.0 if precipitation_amount_cm is None else float(precipitation_amount_cm)

        rh_frac = float(rh1) / 100.0 if float(rh1) > 1.0 else float(rh1)
        if rh_frac < 0.001 or rh_frac > 1.0:
            return False
        if ta1 < -60.0 or ta1 > 60.0:
            return False

        self.m_ta1 = float(ta1)
        self.m_ha1 = rh_frac
        self.m_sv1 = max(0.0, float(sr1)) / self.Smv
        self.m_rc1 = max(0.0, float(rcum_cm))
        self.m_ra1 = self.m_rc1  # prcpAsAmnt = True semantic
        self.m_bp0 = self.m_bp1
        self.m_bp1 = float(barometric_pressure)

        self.m_updates += 1
        self.m_elapsed += et

        if self.m_ra1 < 0.0001:
            self.m_rdur = 0.0
        pptrate = self.m_ra1 / et / self.Pi

        mdt = et / float(self.m_mSteps)
        mdt_2 = mdt * 2.0
        ddt = et / float(self.m_dSteps)

        rai0 = mdt * self.m_rai0 * (1.0 - math.exp(-100.0 * pptrate))
        if self.m_ha1 < self.m_ha0:
            if self.m_rampRai0:
                rai0 *= 1.0 - ((self.m_ha0 - self.m_ha1) / self.m_ha0)
            else:
                rai0 *= 0.15
        rai1 = mdt * self.m_rai1 * pptrate

        tstate = [0] * 11
        ddtNext = ddt
        nstep = 1

        while nstep <= self.m_mSteps:
            tt = nstep * mdt
            tfract = tt / et
            ta = self.m_ta0 + (self.m_ta1 - self.m_ta0) * tfract
            ha = self.m_ha0 + (self.m_ha1 - self.m_ha0) * tfract
            sv = self.m_sv0 + (self.m_sv1 - self.m_sv0) * tfract
            bp = self.m_bp0 + (self.m_bp1 - self.m_bp0) * tfract

            fsc = sv / self.Srf
            tka = ta + self.Kelvin
            tdw = 5205.0 / ((5205.0 / tka) - math.log(ha))
            tdp = tdw - self.Kelvin
            tsk = (self.Tcn + self.Kelvin) if (fsc < 0.000001) else (self.Tcd + self.Kelvin)
            hr = self.Hrn if (fsc < 0.000001) else self.Hrd
            sr = 0.0 if (fsc < 0.000001) else (self.Srf * fsc)

            psa = 0.0000239 * math.exp(20.58 - (5205.0 / tka))
            pa = ha * psa
            psd = 0.0000239 * math.exp(20.58 - (5205.0 / tdw))

            if self.m_ra1 > 0.0001:
                self.m_rdur += mdt
            else:
                self.m_rdur = 0.0

            tfd = ta + (sr - hr * (ta - tsk + self.Kelvin)) / (hr + self.m_hc)
            qv = 13550.0 - 10.22 * (tfd + self.Kelvin)
            hw = (self.m_hwf * self.Ap / 0.24) * qv / 18.0
            self.m_t[0] = tfd - (hw * (tfd - ta) / (hr + self.m_hc + hw))

            qw = 5040.0 * math.exp(-14.0 * self.m_w[0])
            tkf = self.m_t[0] + self.Kelvin
            gnu = 0.00439 + 0.00000177 * math.pow((338.76 - tkf), 2.1237)

            c1 = 0.1617 - 0.001419 * self.m_t[0]
            c2 = 0.4657 + 0.003578 * self.m_t[0]
            self.m_wsa = c1 * math.pow(self.Wsf, c2)
            wdiff = max(1e-6, self.m_wmax - self.m_wsa)

            ps1 = 0.0000239 * math.exp(20.58 - (5205.0 / tkf))
            p1 = max(1e-6, pa + self.Ap * bp * (qv / (qv + qw)) * (tka - tkf))

            self.m_hf = min(self.Hfs, p1 / ps1)
            hf_log = -math.log(1.0 - self.m_hf)
            self.m_sem = c1 * math.pow(hf_log, c2)

            self.m_state = int(NelsonDFMState.NONE)
            self.m_wfilm = 0.0
            s_new = self.m_s[0]
            w_old = self.m_w[0]

            if self.m_ra1 > 0.0:
                if self.m_allowRainstorm and pptrate >= self.m_stv:
                    self.m_state = int(NelsonDFMState.RAINSTORM)
                    self.m_wfilm = self.m_wfilmk
                    w_new = self.m_wmx
                else:
                    if self.m_rdur < 1.0 or not self.m_allowRainfall2:
                        self.m_state = int(NelsonDFMState.RAINFALL1)
                        w_new = w_old + rai0
                    else:
                        self.m_state = int(NelsonDFMState.RAINFALL2)
                        w_new = w_old + rai1
                self.m_wfilm = self.m_wfilmk
                s_new = (w_new - self.m_wsa) / wdiff
                self.m_t[0] = tfd
                self.m_hf = self.Hfs
            else:
                if w_old > self.m_wsa:
                    p1 = ps1
                    self.m_hf = self.Hfs
                    aml = self.m_amlf * (ps1 - psd) / bp
                    if self.m_t[0] <= tdp and p1 > psd:
                        aml = 0.0
                    w_new = w_old - aml * mdt_2
                    if aml > 0.0:
                        w_new -= (mdt * self.m_capf / gnu)
                    if w_new > self.m_wmx:
                        w_new = self.m_wmx
                    s_new = (w_new - self.m_wsa) / wdiff

                    if w_new > w_old:
                        self.m_state = int(NelsonDFMState.CONDENSATION1)
                    elif w_new == w_old:
                        self.m_state = int(NelsonDFMState.STAGNATION)
                    else:
                        self.m_state = int(NelsonDFMState.EVAPORATION)
                elif self.m_t[0] <= tdp:
                    self.m_state = int(NelsonDFMState.CONDENSATION2)
                    aml = 0.0 if (p1 > psd) else (self.m_amlf * (p1 - psd) / bp)
                    w_new = w_old - aml * mdt_2
                    s_new = (w_new - self.m_wsa) / wdiff
                else:
                    if w_old >= self.m_sem:
                        self.m_state = int(NelsonDFMState.DESORPTION)
                        bi = self.m_stcd * self.m_dx / self.m_d[0]
                    else:
                        self.m_state = int(NelsonDFMState.ADSORPTION)
                        bi = self.m_stca * self.m_dx / self.m_d[0]
                    w_new = (self.m_w[1] + bi * self.m_sem) / (1.0 + bi)
                    s_new = 0.0

            self.m_w[0] = min(self.m_wmx, w_new)
            self.m_s[0] = max(0.0, s_new)
            tstate[self.m_state] += 1

            for i in range(self.m_nodes):
                self.m_Twold[i] = self.m_w[i]
                self.m_Tsold[i] = self.m_s[i]
                self.m_Ttold[i] = self.m_t[i]
                self.m_Tv[i] = self.Thdiff * self.m_x[i]
                self.m_To[i] = self.m_d[i] * self.m_x[i]

            if self.m_state != int(NelsonDFMState.STAGNATION):
                for i in range(self.m_nodes):
                    self.m_Tg[i] = 0.0
                    svp = (self.m_w[i] - self.m_wsa) / wdiff
                    if self.Sir <= svp <= self.Scr:
                        ak = self.Aks * (2.0 * math.sqrt(svp / self.Scr) - 1.0)
                        self.m_Tg[i] = (
                            (ak / (gnu * wdiff))
                            * self.m_x[i]
                            * self.m_vf
                            * math.pow((self.Scr / svp), 1.5)
                        )

                for i in range(1, self.m_nodes - 1):
                    ae = self.m_Tg[i + 1] / self.m_dx
                    aw = self.m_Tg[i - 1] / self.m_dx
                    ar = self.m_x[i] * self.m_dx / mdt
                    ap = ae + aw + ar
                    self.m_s[i] = (
                        ae * self.m_Tsold[i + 1]
                        + aw * self.m_Tsold[i - 1]
                        + ar * self.m_Tsold[i]
                    ) / ap
                    self.m_s[i] = max(0.0, min(self.Sir, self.m_s[i]))
                self.m_s[self.m_nodes - 1] = self.m_s[self.m_nodes - 2]

                continuous_liquid = True
                for i in range(1, self.m_nodes - 1):
                    if self.m_s[i] < self.Sir:
                        continuous_liquid = False
                        break

                if continuous_liquid:
                    for i in range(1, self.m_nodes - 1):
                        self.m_w[i] = min(self.m_wmx, max(0.0, self.m_wsa + self.m_s[i] * wdiff))
                else:
                    for i in range(1, self.m_nodes - 1):
                        ae = self.m_To[i + 1] / self.m_dx
                        aw = self.m_To[i - 1] / self.m_dx
                        ar = self.m_x[i] * self.m_dx / mdt
                        ap = ae + aw + ar
                        w_calc = (
                            ae * self.m_Twold[i + 1]
                            + aw * self.m_Twold[i - 1]
                            + ar * self.m_Twold[i]
                        ) / ap
                        self.m_w[i] = min(self.m_wmx, max(0.0, w_calc))
                self.m_w[self.m_nodes - 1] = self.m_w[self.m_nodes - 2]

            # Thermal diffusion update
            for i in range(1, self.m_nodes - 1):
                ae = self.m_Tv[i + 1] / self.m_dx
                aw = self.m_Tv[i - 1] / self.m_dx
                ar = self.m_x[i] * self.m_dx / mdt
                ap = ae + aw + ar
                t_calc = (
                    ae * self.m_Ttold[i + 1]
                    + aw * self.m_Ttold[i - 1]
                    + ar * self.m_Ttold[i]
                ) / ap
                self.m_t[i] = min(71.0, t_calc)
            self.m_t[self.m_nodes - 1] = self.m_t[self.m_nodes - 2]

            # Diffusivity sub-stepping trigger
            if (ddtNext - tt) < (0.5 * mdt):
                self._calculate_diffusivity(bp)
                ddtNext += ddt

            nstep += 1

        # Dominant state over the interval
        self.m_state = int(NelsonDFMState.NONE)
        max_cnt = tstate[0]
        for i in range(1, 11):
            if tstate[i] > max_cnt:
                self.m_state = i
                max_cnt = tstate[i]

        return True

    # --------------------------------------------------------------------------
    # Output Getters & Diagnostics
    # --------------------------------------------------------------------------
    def get_moisture(self) -> float:
        """
        Returns the mean volume-weighted moisture content as dimensionless fraction (g/g).
        """
        wbr = sum(self.m_w[i] * self.m_v[i] for i in range(self.m_nodes))
        wbr = min(self.m_wmx, wbr) + self.m_wfilm
        return float(wbr)

    def get_moisture_percent(self) -> float:
        """
        Returns the mean volume-weighted moisture content as percentage (%).
        """
        return moisture_ratio_to_percent(self.get_moisture())

    def get_temperature(self) -> float:
        """
        Returns the mean volume-weighted temperature across the cylinder in °C.
        """
        return float(sum(self.m_t[i] * self.m_v[i] for i in range(self.m_nodes)))

    def get_surface_moisture(self) -> float:
        """Returns the outer surface node moisture fraction (g/g)."""
        return float(self.m_w[0])

    def get_surface_moisture_percent(self) -> float:
        """Returns the outer surface node moisture percentage (%)."""
        return moisture_ratio_to_percent(self.get_surface_moisture())

    def get_surface_temperature(self) -> float:
        """Returns the outer surface node temperature in °C."""
        return float(self.m_t[0])

    def get_median_moisture(self) -> float:
        """Returns the radial median moisture fraction (g/g)."""
        s = sorted(self.m_w)
        return float(s[len(s) // 2])

    def get_state_id(self) -> int:
        """Returns the current dominant DFM state ID."""
        return int(self.m_state)

    def get_state_name(self) -> str:
        """Returns the human-readable name of the current DFM state."""
        return NELSON_STATE_NAMES[self.m_state]

    def get_state(self) -> NelsonFuelStickState:
        """
        Returns a complete, immutable snapshot of the fuel stick state.
        """
        m_frac = self.get_moisture()
        med_frac = self.get_median_moisture()
        surf_frac = self.get_surface_moisture()

        return NelsonFuelStickState(
            elapsed_hours=self.m_elapsed,
            updates_count=self.m_updates,
            mean_weighted_moisture=m_frac,
            mean_weighted_moisture_pct=m_frac * 100.0,
            median_radial_moisture=med_frac,
            median_radial_moisture_pct=med_frac * 100.0,
            surface_moisture=surf_frac,
            surface_moisture_pct=surf_frac * 100.0,
            surface_temperature_c=self.get_surface_temperature(),
            mean_weighted_temperature_c=self.get_temperature(),
            surface_film=self.m_wfilm,
            state_id=self.get_state_id(),
            state_name=self.get_state_name(),
            moisture_profile=tuple(self.m_w),
            temperature_profile=tuple(self.m_t),
            diffusivity_profile=tuple(self.m_d),
            saturation_profile=tuple(self.m_s),
        )

    # --------------------------------------------------------------------------
    # Helper utilities
    # --------------------------------------------------------------------------
    @staticmethod
    def _extract_attr(obj: Any, key: str, default: float) -> float:
        if isinstance(obj, dict):
            return float(obj.get(key, default))
        return float(getattr(obj, key, default))


# ==============================================================================
# CONTINUOUS-TIME FUEL MOISTURE PROVIDER ADAPTER (NELSON-002B)
# ==============================================================================

class NelsonFuelMoistureProvider(FuelMoistureProvider):
    """
    Continuous-time, query-order independent FuelMoistureProvider adapter
    powered by the Nelson (2000) Physical Dead Fuel Moisture Model.

    Maintains 3 independent cylindrical sticks (1-h, 10-h, 100-h) and constant live
    fractions (live_herbaceous, live_woody).
    Precomputes frozen hourly boundary checkpoints at t = 0, 60, 120, ... min,
    allowing O(1) retrieval at integer hours and isolated partial-step physical
    integration for arbitrary sub-hourly timestamps.
    """

    def __init__(
        self,
        weather_manager: Any = None,
        hourly_records: Optional[List[Dict[str, Any]]] = None,
        conditioning_records: Optional[List[Dict[str, Any]]] = None,
        initialization_mode: str = "explicit",  # "explicit" | "conditioned"
        seed_mode: str = "nfdrs_reference",     # "nfdrs_reference" | "explicit"
        initial_fuel_moisture: Optional[FuelMoistureState] = None,
        live_herbaceous: float = 0.60,
        live_woody: float = 0.90,
        seed_dead_1h: float = 0.20,
        seed_dead_10h: float = 0.20,
        seed_dead_100h: float = 0.20,
        seed_temperature_c: float = 20.0,
        barometric_pressure: float = 0.0218,
        conditioning_hours: Optional[int] = None,
        conditioning_start_datetime_utc: Optional[str] = None,
        lookup_mode: str = "exact_subhour",  # "exact_subhour" | "dense_timeseries"
        dense_dt_minutes: float = 15.0,
        dense_interpolation_method: str = "linear",  # "linear" | "previous"
        simulation_duration_minutes: Optional[float] = None,
    ):
        self._live_herbaceous = float(live_herbaceous)
        self._live_woody = float(live_woody)
        self._barometric_pressure = float(barometric_pressure)
        self._initialization_mode = initialization_mode.lower().strip()
        self._seed_mode = seed_mode.lower().strip()
        mode_norm = lookup_mode.lower().strip()
        from fire_engine.schemas import VALID_NELSON_RUNTIME_MODES
        valid_modes = set(VALID_NELSON_RUNTIME_MODES) | {"exact_subhour"}
        if mode_norm not in valid_modes:
            raise ValueError(
                f"Unknown lookup_mode '{lookup_mode}' for NelsonFuelMoistureProvider. "
                f"Valid modes: {sorted(list(valid_modes))}"
            )
        self._lookup_mode = mode_norm
        self._dense_dt_minutes = float(dense_dt_minutes)
        self._dense_interpolation_method = dense_interpolation_method.lower().strip()
        self._simulation_duration_minutes = simulation_duration_minutes
        self._dense_timeseries: Optional[NelsonFuelMoistureTimeSeries] = None
        self._physical_runtime_fallback_count: int = 0
        self.conditioning_report: Optional[NelsonConditioningReport] = None

        # Resolve simulation records and conditioning records
        if hourly_records is not None:
            self._records = list(hourly_records)
            self._cond_records = list(conditioning_records or [])
        elif weather_manager is not None:
            if hasattr(weather_manager, "get_simulation_records"):
                self._records = weather_manager.get_simulation_records()
            elif hasattr(weather_manager, "hourly_records"):
                self._records = weather_manager.hourly_records
            else:
                raise NelsonError("WeatherManager provided does not contain hourly records.")

            if self._initialization_mode == "conditioned":
                if conditioning_records is not None:
                    self._cond_records = list(conditioning_records)
                elif hasattr(weather_manager, "get_conditioning_records"):
                    self._cond_records = weather_manager.get_conditioning_records(
                        conditioning_hours=conditioning_hours,
                        conditioning_start_datetime_utc=conditioning_start_datetime_utc,
                    )
                else:
                    self._cond_records = []
            else:
                self._cond_records = []
        else:
            raise NelsonError(
                "NelsonFuelMoistureProvider requires either a WeatherManager or an explicit list of hourly_records."
            )

        if not self._records:
            raise MissingWeatherRecordError("Simulation hourly weather records list is empty.")

        self._num_hours = len(self._records)

        # Validate meteorological inputs for Nelson physics
        self._validate_weather_records(self._records, context="simulation")
        if self._cond_records:
            self._validate_weather_records(self._cond_records, context="conditioning")

        # Initialize the 3 stick models
        self._m1 = NelsonPhysicalModel.for_1h()
        self._m10 = NelsonPhysicalModel.for_10h()
        self._m100 = NelsonPhysicalModel.for_100h()

        # Handle initial seed values
        if self._initialization_mode == "conditioned":
            if self._seed_mode == "nfdrs_reference":
                s1, s10, s100 = 0.20, 0.20, 0.20
                st = 20.0
            elif self._seed_mode == "explicit":
                s1, s10, s100 = seed_dead_1h, seed_dead_10h, seed_dead_100h
                st = seed_temperature_c
            else:
                raise ValueError(
                    f"Unsupported seed_mode for Nelson conditioning: '{seed_mode}'. Use 'nfdrs_reference' or 'explicit'."
                )

            self._m1.initialize(init_moisture=s1, init_temp_c=st, barometric_pressure=self._barometric_pressure)
            self._m10.initialize(init_moisture=s10, init_temp_c=st, barometric_pressure=self._barometric_pressure)
            self._m100.initialize(init_moisture=s100, init_temp_c=st, barometric_pressure=self._barometric_pressure)

            self._run_conditioning(s1, s10, s100, weather_manager)
        else:
            if initial_fuel_moisture is not None:
                s1 = initial_fuel_moisture.dead_1h
                s10 = initial_fuel_moisture.dead_10h
                s100 = initial_fuel_moisture.dead_100h
                st = seed_temperature_c
            elif self._seed_mode == "explicit":
                s1, s10, s100 = seed_dead_1h, seed_dead_10h, seed_dead_100h
                st = seed_temperature_c
            else:
                s1, s10, s100 = 0.20, 0.20, 0.20
                st = 20.0

            self._m1.initialize(init_moisture=s1, init_temp_c=st, barometric_pressure=self._barometric_pressure)
            self._m10.initialize(init_moisture=s10, init_temp_c=st, barometric_pressure=self._barometric_pressure)
            self._m100.initialize(init_moisture=s100, init_temp_c=st, barometric_pressure=self._barometric_pressure)

        # Precompute frozen hourly boundary checkpoints
        self._precompute_checkpoints()

        # If table mode requested at init, precompute dense table
        if self._lookup_mode == "dense_timeseries":
            self._dense_timeseries = self.build_dense_timeseries(
                dt_minutes=self._dense_dt_minutes,
                interpolation_method=self._dense_interpolation_method,
                simulation_duration_minutes=self._simulation_duration_minutes,
            )
        elif self._lookup_mode in ("exact_sampled", "exact_sampled_5min", "hourly_anchored_exact_sampled_5min"):
            self._dense_timeseries = self.build_exact_sampled_timeseries(
                dt_minutes=5.0,
                interpolation_method="linear",
                simulation_duration_minutes=self._simulation_duration_minutes,
            )
        elif self._lookup_mode == "nfdrs4_sequential_5min":
            self._dense_timeseries = self.build_dense_timeseries(
                dt_minutes=5.0,
                interpolation_method="linear",
                simulation_duration_minutes=self._simulation_duration_minutes,
            )
        elif self._lookup_mode == "adaptive_exact_sampled":
            self._dense_timeseries = self.build_adaptive_exact_sampled_timeseries(
                simulation_duration_minutes=self._simulation_duration_minutes,
            )

    def _validate_weather_records(self, records: List[Dict[str, Any]], context: str) -> None:
        """
        Validates presence and ranges of required meteorological fields for Nelson physics.
        """
        for idx, rec in enumerate(records):
            if "solar_radiation_w_m2" not in rec or rec["solar_radiation_w_m2"] is None:
                raise NelsonRequiredWeatherInputMissingError(
                    f"NELSON_REQUIRED_WEATHER_INPUT_MISSING: 'solar_radiation_w_m2' is required in {context} "
                    f"record index {idx}, but got None or missing."
                )
            sol = rec["solar_radiation_w_m2"]
            try:
                sol_f = float(sol)
                if math.isnan(sol_f) or math.isinf(sol_f) or sol_f < 0.0:
                    raise NelsonError(f"Invalid solar_radiation_w_m2 value '{sol}' in {context} record {idx}.")
            except (ValueError, TypeError):
                raise NelsonError(f"Non-numeric solar_radiation_w_m2 '{sol}' in {context} record {idx}.")

            for req in ("temperature_c", "relative_humidity_pct"):
                if req not in rec or rec[req] is None:
                    raise MissingWeatherRecordError(f"Field '{req}' missing in {context} record {idx}.")
                try:
                    v = float(rec[req])
                    if math.isnan(v) or math.isinf(v):
                        raise NelsonError(f"Invalid {req} value in {context} record {idx}: {v}")
                except (ValueError, TypeError):
                    raise NelsonError(f"Non-numeric {req} in {context} record {idx}.")

    def _run_conditioning(self, s1: float, s10: float, s100: float, weather_manager: Any) -> None:
        """
        Chronologically executes Nelson physical models across conditioning records [-N, 0).
        """
        prev_wx = None
        for idx, rec in enumerate(self._cond_records):
            curr_wx = {
                "temperature_c": float(rec["temperature_c"]),
                "relative_humidity_pct": float(rec["relative_humidity_pct"]),
                "solar_radiation_w_m2": float(rec["solar_radiation_w_m2"]),
                "precipitation_mm": float(rec.get("precipitation_mm", 0.0)),
            }
            self._m1.update(weather_previous=prev_wx, weather_current=curr_wx, dt_hours=1.0)
            self._m10.update(weather_previous=prev_wx, weather_current=curr_wx, dt_hours=1.0)
            self._m100.update(weather_previous=prev_wx, weather_current=curr_wx, dt_hours=1.0)
            prev_wx = curr_wx

        wx_start = getattr(weather_manager, "start_datetime_utc", None) if weather_manager else None
        sim_start = getattr(weather_manager, "simulation_start_datetime_utc", None) if weather_manager else None
        cond_start = self._cond_records[0].get("datetime_utc") if self._cond_records else None

        self.conditioning_report = NelsonConditioningReport(
            conditioning_hours=len(self._cond_records),
            weather_records_used=len(self._cond_records),
            seed_mode=self._seed_mode,
            seed_dead_1h=s1,
            seed_dead_10h=s10,
            seed_dead_100h=s100,
            final_dead_1h=self._m1.get_moisture(),
            final_dead_10h=self._m10.get_moisture(),
            final_dead_100h=self._m100.get_moisture(),
            final_temperature_1h_c=self._m1.get_temperature(),
            final_temperature_10h_c=self._m10.get_temperature(),
            final_temperature_100h_c=self._m100.get_temperature(),
            weather_data_start_datetime_utc=wx_start,
            simulation_start_datetime_utc=sim_start,
            conditioning_start_datetime_utc=cond_start,
        )

    def _precompute_checkpoints(self) -> None:
        """
        Chronologically integrates and freezes Nelson stick checkpoints at each integer hour:
        checkpoint[0] = state at t = 0 min (post-conditioning or initial seed)
        checkpoint[h+1] = state at t = (h+1)*60 min after applying simulation_record[h]
        """
        fmc0 = FuelMoistureState(
            dead_1h=self._m1.get_moisture(),
            dead_10h=self._m10.get_moisture(),
            dead_100h=self._m100.get_moisture(),
            live_herbaceous=self._live_herbaceous,
            live_woody=self._live_woody,
        )
        chk0 = NelsonTriStickCheckpoint(
            elapsed_minutes=0.0,
            hour_index=0,
            chk_1h=self._m1.get_checkpoint(),
            chk_10h=self._m10.get_checkpoint(),
            chk_100h=self._m100.get_checkpoint(),
            fuel_moisture_state=fmc0,
            weather_record=None,
        )
        checkpoints = [chk0]

        prev_wx = None
        for h, rec in enumerate(self._records):
            curr_wx = {
                "temperature_c": float(rec["temperature_c"]),
                "relative_humidity_pct": float(rec["relative_humidity_pct"]),
                "solar_radiation_w_m2": float(rec["solar_radiation_w_m2"]),
                "precipitation_mm": float(rec.get("precipitation_mm", 0.0)),
            }
            self._m1.update(weather_previous=prev_wx, weather_current=curr_wx, dt_hours=1.0)
            self._m10.update(weather_previous=prev_wx, weather_current=curr_wx, dt_hours=1.0)
            self._m100.update(weather_previous=prev_wx, weather_current=curr_wx, dt_hours=1.0)
            prev_wx = curr_wx

            fmc_h = FuelMoistureState(
                dead_1h=self._m1.get_moisture(),
                dead_10h=self._m10.get_moisture(),
                dead_100h=self._m100.get_moisture(),
                live_herbaceous=self._live_herbaceous,
                live_woody=self._live_woody,
            )
            chk_h = NelsonTriStickCheckpoint(
                elapsed_minutes=(h + 1) * 60.0,
                hour_index=h + 1,
                chk_1h=self._m1.get_checkpoint(),
                chk_10h=self._m10.get_checkpoint(),
                chk_100h=self._m100.get_checkpoint(),
                fuel_moisture_state=fmc_h,
                weather_record=rec,
            )
            checkpoints.append(chk_h)

        self._hour_checkpoints: List[NelsonTriStickCheckpoint] = checkpoints

    @property
    def initial_state(self) -> FuelMoistureState:
        """Returns the initial state at t=0 min (checkpoint 0)."""
        return self._hour_checkpoints[0].fuel_moisture_state

    @property
    def hour_checkpoints(self) -> List[NelsonTriStickCheckpoint]:
        """Returns the list of frozen hourly checkpoints."""
        return self._hour_checkpoints

    @property
    def num_hours(self) -> int:
        """Returns number of hourly simulation weather records."""
        return self._num_hours

    @property
    def physical_runtime_fallback_count(self) -> int:
        """Returns count of physical fallback evaluations during runtime lookup (strictly 0 in table mode)."""
        return self._physical_runtime_fallback_count

    @property
    def nelson_runtime_physical_calls(self) -> int:
        """Alias for physical_runtime_fallback_count required by NELSON-003A instrumentation."""
        return self._physical_runtime_fallback_count

    def is_time_dependent(self) -> bool:
        """Indicates this provider evolves dynamically with time."""
        return True

    def is_spatially_variable(self) -> bool:
        """Indicates this provider delivers uniform spatial values for now."""
        return False

    def get_state(
        self,
        elapsed_minutes: float = 0.0,
        row: Optional[int] = None,
        col: Optional[int] = None,
    ) -> FuelMoistureState:
        """
        Returns the FuelMoistureState for any continuous elapsed_minutes in [0, max_minutes].

        - When lookup_mode in ('dense_timeseries', 'exact_sampled', 'exact_sampled_5min',
          'hourly_anchored_exact_sampled_5min', 'nfdrs4_sequential_5min', 'adaptive_exact_sampled'):
          delegates strictly to precomputed lookup table (zero physical ODE fallback).
        - When lookup_mode == 'exact_subhour': evaluates exact physical ODE integration.
        - Guaranteed deterministic and query-order independent.
        """
        t = float(elapsed_minutes)
        if t < 0.0:
            raise WeatherRangeError(f"Negative simulation time not allowed: {t} min.")

        max_minutes = self._num_hours * 60.0
        if t > max_minutes:
            raise WeatherRangeError(
                f"Requested simulation time {t:.2f} min exceeds available weather range ({max_minutes:.1f} min)."
            )

        # Opt-in fast runtime table path (zero physical ODE integration during runtime)
        if self._lookup_mode in (
            "dense_timeseries",
            "exact_sampled",
            "exact_sampled_5min",
            "hourly_anchored_exact_sampled_5min",
            "nfdrs4_sequential_5min",
            "adaptive_exact_sampled",
        ) and self._dense_timeseries is not None:
            return self._dense_timeseries.get_state_fast(t, row, col)

        return self._get_state_exact_subhour(t)

    def get_state_exact(
        self,
        elapsed_minutes: float = 0.0,
        row: Optional[int] = None,
        col: Optional[int] = None,
    ) -> FuelMoistureState:
        """
        Forces physical sub-hourly ODE integration from base hour checkpoint,
        bypassing dense timeseries table even if lookup_mode='dense_timeseries'.
        Guaranteed deterministic and query-order independent.
        """
        t = float(elapsed_minutes)
        if t < 0.0:
            raise WeatherRangeError(f"Negative simulation time not allowed: {t} min.")
        max_minutes = self._num_hours * 60.0
        if t > max_minutes:
            raise WeatherRangeError(
                f"Requested simulation time {t:.2f} min exceeds available weather range ({max_minutes:.1f} min)."
            )
        return self._get_state_exact_subhour(t)

    def _get_state_exact_subhour(self, t: float) -> FuelMoistureState:
        """Internal physical ODE integration from base hour checkpoint."""
        h = int(math.floor(t / 60.0))
        remainder_min = t - (h * 60.0)
        max_minutes = self._num_hours * 60.0

        # Exact hour boundary (O(1) direct retrieval)
        if math.isclose(remainder_min, 0.0, abs_tol=1e-9):
            return self._hour_checkpoints[h].fuel_moisture_state
        if h >= self._num_hours and math.isclose(t, max_minutes, abs_tol=1e-9):
            return self._hour_checkpoints[-1].fuel_moisture_state

        # Partial-hour physical integration
        base_chk = self._hour_checkpoints[h]
        rec = self._records[h]
        f = remainder_min / 60.0

        # Atmospheric linear interpolation across the hourly interval [h, h+1)
        t_start = base_chk.chk_10h.ta1
        rh_start = base_chk.chk_10h.ha1 * 100.0
        sr_start = base_chk.chk_10h.sv1 * 94.743

        t_target = float(rec["temperature_c"])
        rh_target = float(rec["relative_humidity_pct"])
        sr_target = float(rec["solar_radiation_w_m2"])
        p_target = float(rec.get("precipitation_mm", 0.0))

        interp_wx = {
            "temperature_c": t_start + f * (t_target - t_start),
            "relative_humidity_pct": rh_start + f * (rh_target - rh_start),
            "solar_radiation_w_m2": sr_start + f * (sr_target - sr_start),
            "precipitation_mm": f * p_target,
        }

        # Advance independent clones from the base checkpoint
        c1 = NelsonPhysicalModel.from_checkpoint(base_chk.chk_1h)
        c10 = NelsonPhysicalModel.from_checkpoint(base_chk.chk_10h)
        c100 = NelsonPhysicalModel.from_checkpoint(base_chk.chk_100h)

        c1.update(weather_current=interp_wx, dt_hours=f)
        c10.update(weather_current=interp_wx, dt_hours=f)
        c100.update(weather_current=interp_wx, dt_hours=f)

        return FuelMoistureState(
            dead_1h=c1.get_moisture(),
            dead_10h=c10.get_moisture(),
            dead_100h=c100.get_moisture(),
            live_herbaceous=self._live_herbaceous,
            live_woody=self._live_woody,
        )

    def get_stick_state(
        self,
        stick_class: str,
        elapsed_minutes: float = 0.0,
    ) -> NelsonFuelStickState:
        """
        Returns the full physical state (radial profiles, temperature, surface film, DFM state)
        for an individual fuel stick at any continuous timestamp.
        """
        sc = stick_class.lower().strip()
        t = float(elapsed_minutes)
        max_minutes = self._num_hours * 60.0
        if t < 0.0 or t > max_minutes:
            raise WeatherRangeError(f"Requested time {t} min exceeds range [0, {max_minutes}].")

        h = int(math.floor(t / 60.0))
        remainder_min = t - (h * 60.0)

        base_chk = self._hour_checkpoints[h]
        if sc == "1h":
            chk = base_chk.chk_1h
        elif sc == "10h":
            chk = base_chk.chk_10h
        elif sc == "100h":
            chk = base_chk.chk_100h
        else:
            raise ValueError(f"Unknown stick_class '{stick_class}'. Use '1h', '10h', or '100h'.")

        if math.isclose(remainder_min, 0.0, abs_tol=1e-9):
            model = NelsonPhysicalModel.from_checkpoint(chk)
            return model.get_state()
        if h >= self._num_hours and math.isclose(t, max_minutes, abs_tol=1e-9):
            model = NelsonPhysicalModel.from_checkpoint(chk)
            return model.get_state()

        rec = self._records[h]
        f = remainder_min / 60.0
        t_start = chk.ta1
        rh_start = chk.ha1 * 100.0
        sr_start = chk.sv1 * 94.743

        interp_wx = {
            "temperature_c": t_start + f * (float(rec["temperature_c"]) - t_start),
            "relative_humidity_pct": rh_start + f * (float(rec["relative_humidity_pct"]) - rh_start),
            "solar_radiation_w_m2": sr_start + f * (float(rec["solar_radiation_w_m2"]) - sr_start),
            "precipitation_mm": f * float(rec.get("precipitation_mm", 0.0)),
        }

        model = NelsonPhysicalModel.from_checkpoint(chk)
        model.update(weather_current=interp_wx, dt_hours=f)
        return model.get_state()

    @property
    def lookup_mode(self) -> str:
        """Returns current lookup mode ('exact_subhour' or 'dense_timeseries')."""
        return self._lookup_mode

    @property
    def dense_timeseries(self) -> Optional[NelsonFuelMoistureTimeSeries]:
        """Returns the precomputed dense timeseries lookup table if initialized."""
        return self._dense_timeseries

    def get_state_fast(
        self,
        elapsed_minutes: float = 0.0,
        row: Optional[int] = None,
        col: Optional[int] = None,
    ) -> FuelMoistureState:
        """
        Fast path query delegating to the precomputed dense time series table.
        If not already initialized, builds it using default parameters.
        """
        if self._dense_timeseries is None:
            self._dense_timeseries = self.build_dense_timeseries(
                dt_minutes=self._dense_dt_minutes,
                interpolation_method=self._dense_interpolation_method,
                simulation_duration_minutes=self._simulation_duration_minutes,
            )
        return self._dense_timeseries.get_state_fast(elapsed_minutes, row, col)

    def enable_dense_timeseries(
        self,
        dt_minutes: float = 15.0,
        interpolation_method: str = "linear",
        simulation_duration_minutes: Optional[float] = None,
    ) -> NelsonFuelMoistureTimeSeries:
        """
        Precomputes the dense runtime lookup table and activates lookup_mode="dense_timeseries".
        """
        self._dense_dt_minutes = float(dt_minutes)
        self._dense_interpolation_method = interpolation_method.lower().strip()
        self._simulation_duration_minutes = simulation_duration_minutes
        self._dense_timeseries = self.build_dense_timeseries(
            dt_minutes=self._dense_dt_minutes,
            interpolation_method=self._dense_interpolation_method,
            simulation_duration_minutes=self._simulation_duration_minutes,
        )
        self._lookup_mode = "dense_timeseries"
        return self._dense_timeseries

    def build_dense_timeseries(
        self,
        dt_minutes: float = 15.0,
        interpolation_method: str = "linear",
        simulation_duration_minutes: Optional[float] = None,
    ) -> NelsonFuelMoistureTimeSeries:
        """
        Chronologically precomputes the Nelson FMC dense time series at resolution dt_minutes.
        Integrates forward step-by-step preserving the full physical state (radial profiles, T, film, D, DFM)
        without repeating integrations.
        Handles simulation durations that are not multiples of dt_minutes.
        """
        dt_min = float(dt_minutes)
        if dt_min <= 0.0:
            raise ValueError(f"dt_minutes must be > 0, got {dt_min}")

        max_allowed_min = self._num_hours * 60.0
        if simulation_duration_minutes is not None:
            total_duration_min = min(float(simulation_duration_minutes), max_allowed_min)
            if total_duration_min < 0.0:
                raise ValueError("simulation_duration_minutes cannot be negative.")
        else:
            total_duration_min = max_allowed_min

        # Clone models from checkpoint t=0 (post-conditioning or initial seed)
        m1 = NelsonPhysicalModel.from_checkpoint(self._hour_checkpoints[0].chk_1h)
        m10 = NelsonPhysicalModel.from_checkpoint(self._hour_checkpoints[0].chk_10h)
        m100 = NelsonPhysicalModel.from_checkpoint(self._hour_checkpoints[0].chk_100h)

        times_list = [0.0]
        d1_list = [m1.get_moisture()]
        d10_list = [m10.get_moisture()]
        d100_list = [m100.get_moisture()]

        t_curr = 0.0
        while t_curr < total_duration_min - 1e-9:
            t_target = min(t_curr + dt_min, total_duration_min)

            # Step forward from t_curr to t_target, splitting at hour boundaries if needed
            t_sub = t_curr
            while t_sub < t_target - 1e-9:
                h = int(t_sub // 60.0)
                if h >= self._num_hours:
                    break
                h_end = float((h + 1) * 60.0)
                t_sub_next = min(t_target, h_end)
                step_dt_min = t_sub_next - t_sub
                step_dt_hours = step_dt_min / 60.0

                base_chk = self._hour_checkpoints[h]
                rec = self._records[h]
                f_next = (t_sub_next - (h * 60.0)) / 60.0

                t_start = base_chk.chk_10h.ta1
                rh_start = base_chk.chk_10h.ha1 * 100.0
                sr_start = base_chk.chk_10h.sv1 * 94.743

                t_target_h = float(rec["temperature_c"])
                rh_target_h = float(rec["relative_humidity_pct"])
                sr_target_h = float(rec["solar_radiation_w_m2"])
                p_total_h = float(rec.get("precipitation_mm", 0.0))

                # Rain amount in this specific sub-interval (conserving sum = p_total_h):
                p_sub = p_total_h * (step_dt_min / 60.0)

                sub_wx = {
                    "temperature_c": t_start + f_next * (t_target_h - t_start),
                    "relative_humidity_pct": rh_start + f_next * (rh_target_h - rh_start),
                    "solar_radiation_w_m2": sr_start + f_next * (sr_target_h - sr_start),
                    "precipitation_mm": p_sub,
                }

                m1.update(weather_current=sub_wx, dt_hours=step_dt_hours)
                m10.update(weather_current=sub_wx, dt_hours=step_dt_hours)
                m100.update(weather_current=sub_wx, dt_hours=step_dt_hours)

                t_sub = t_sub_next

            t_curr = t_target
            times_list.append(t_curr)
            d1_list.append(m1.get_moisture())
            d10_list.append(m10.get_moisture())
            d100_list.append(m100.get_moisture())

        times_arr = np.ascontiguousarray(times_list, dtype=np.float64)
        d1_arr = np.ascontiguousarray(d1_list, dtype=np.float64)
        d10_arr = np.ascontiguousarray(d10_list, dtype=np.float64)
        d100_arr = np.ascontiguousarray(d100_list, dtype=np.float64)

        times_arr.flags.writeable = False
        d1_arr.flags.writeable = False
        d10_arr.flags.writeable = False
        d100_arr.flags.writeable = False

        return NelsonFuelMoistureTimeSeries(
            times_minutes=times_arr,
            dead_1h=d1_arr,
            dead_10h=d10_arr,
            dead_100h=d100_arr,
            live_herbaceous=self._live_herbaceous,
            live_woody=self._live_woody,
            dt_minutes=dt_min,
            total_duration_minutes=total_duration_min,
            interpolation_method=interpolation_method,
            is_uniform=True,
        )

    def enable_exact_sampled_timeseries(
        self,
        dt_minutes: float = 5.0,
        interpolation_method: str = "linear",
        simulation_duration_minutes: Optional[float] = None,
    ) -> NelsonFuelMoistureTimeSeries:
        """
        Precomputes an exact-sampled runtime lookup table and activates lookup_mode="exact_sampled".
        """
        self._dense_dt_minutes = float(dt_minutes)
        self._dense_interpolation_method = interpolation_method.lower().strip()
        self._simulation_duration_minutes = simulation_duration_minutes
        self._dense_timeseries = self.build_exact_sampled_timeseries(
            dt_minutes=self._dense_dt_minutes,
            interpolation_method=self._dense_interpolation_method,
            simulation_duration_minutes=self._simulation_duration_minutes,
        )
        self._lookup_mode = "exact_sampled"
        return self._dense_timeseries

    def build_exact_sampled_timeseries(
        self,
        dt_minutes: float = 5.0,
        interpolation_method: str = "linear",
        simulation_duration_minutes: Optional[float] = None,
    ) -> NelsonFuelMoistureTimeSeries:
        """
        Precomputes an exact-sampled runtime lookup table where all sub-hourly nodes
        are physically integrated directly from the parent hourly checkpoint.
        Eliminates sequential timestep-path accumulation drift.
        """
        dt_min = float(dt_minutes)
        if dt_min <= 0.0:
            raise ValueError(f"dt_minutes must be > 0, got {dt_min}")

        max_allowed_min = self._num_hours * 60.0
        if simulation_duration_minutes is not None:
            total_duration_min = min(float(simulation_duration_minutes), max_allowed_min)
            if total_duration_min < 0.0:
                raise ValueError("simulation_duration_minutes cannot be negative.")
        else:
            total_duration_min = max_allowed_min

        # Generate sample times: [0, dt, 2dt, ..., total_duration_min]
        n_steps = int(math.floor(total_duration_min / dt_min))
        times_list = [round(i * dt_min, 6) for i in range(n_steps + 1)]
        if times_list[-1] < total_duration_min - 1e-9:
            times_list.append(round(total_duration_min, 6))

        d1_list: List[float] = []
        d10_list: List[float] = []
        d100_list: List[float] = []

        for t in times_list:
            s = self.get_state_exact(t)
            d1_list.append(s.dead_1h)
            d10_list.append(s.dead_10h)
            d100_list.append(s.dead_100h)

        times_arr = np.ascontiguousarray(times_list, dtype=np.float64)
        d1_arr = np.ascontiguousarray(d1_list, dtype=np.float64)
        d10_arr = np.ascontiguousarray(d10_list, dtype=np.float64)
        d100_arr = np.ascontiguousarray(d100_list, dtype=np.float64)

        times_arr.flags.writeable = False
        d1_arr.flags.writeable = False
        d10_arr.flags.writeable = False
        d100_arr.flags.writeable = False

        return NelsonFuelMoistureTimeSeries(
            times_minutes=times_arr,
            dead_1h=d1_arr,
            dead_10h=d10_arr,
            dead_100h=d100_arr,
            live_herbaceous=self._live_herbaceous,
            live_woody=self._live_woody,
            dt_minutes=dt_min,
            total_duration_minutes=total_duration_min,
            interpolation_method=interpolation_method,
            is_uniform=True,
        )

    def enable_adaptive_exact_sampled_timeseries(
        self,
        fmc_error_tol: float = 0.0005,
        min_dt_minutes: float = 1.0,
        max_dt_minutes: float = 15.0,
        target_mx_list: Optional[List[float]] = None,
        interpolation_method: str = "linear",
        simulation_duration_minutes: Optional[float] = None,
    ) -> NelsonFuelMoistureTimeSeries:
        """
        Precomputes an adaptive exact-sampled runtime lookup table and activates lookup_mode="adaptive_exact_sampled".
        """
        self._dense_dt_minutes = float(max_dt_minutes)
        self._dense_interpolation_method = interpolation_method.lower().strip()
        self._simulation_duration_minutes = simulation_duration_minutes
        self._dense_timeseries = self.build_adaptive_exact_sampled_timeseries(
            fmc_error_tol=fmc_error_tol,
            min_dt_minutes=min_dt_minutes,
            max_dt_minutes=max_dt_minutes,
            target_mx_list=target_mx_list,
            interpolation_method=self._dense_interpolation_method,
            simulation_duration_minutes=self._simulation_duration_minutes,
        )
        self._lookup_mode = "adaptive_exact_sampled"
        return self._dense_timeseries

    def build_adaptive_exact_sampled_timeseries(
        self,
        fmc_error_tol: float = 0.0005,
        min_dt_minutes: float = 1.0,
        max_dt_minutes: float = 15.0,
        target_mx_list: Optional[List[float]] = None,
        interpolation_method: str = "linear",
        simulation_duration_minutes: Optional[float] = None,
    ) -> NelsonFuelMoistureTimeSeries:
        """
        Constructs an adaptive exact-sampled runtime lookup table using recursive interval refinement during precompute.
        All sample nodes are generated strictly from the parent hourly checkpoint reference semantics (exact_subhour),
        completely eliminating sequential timestep drift.

        Refinement triggers:
        1. FMC linear interpolation error at midpoint exceeding fmc_error_tol for 1h or 10h stick.
        2. Extinction threshold crossing: if [ta, tb] crosses any target Mx.
        3. Extinction proximity: if midpoint is within 0.003 of any target Mx and (tb - ta) > 2 * min_dt_minutes.
        """
        if target_mx_list is None:
            target_mx_list = [0.12, 0.15, 0.20, 0.25, 0.30]

        max_allowed_min = self._num_hours * 60.0
        if simulation_duration_minutes is not None:
            total_duration_min = min(float(simulation_duration_minutes), max_allowed_min)
            if total_duration_min < 0.0:
                raise ValueError("simulation_duration_minutes cannot be negative.")
        else:
            total_duration_min = max_allowed_min

        states: Dict[float, FuelMoistureState] = {}
        def get_s(t: float) -> FuelMoistureState:
            t_r = round(t, 6)
            if t_r not in states:
                states[t_r] = self.get_state_exact(t_r)
            return states[t_r]

        num_hours = int(math.ceil(total_duration_min / 60.0))
        for h in range(num_hours):
            h_start = float(h * 60.0)
            h_end = min(float((h + 1) * 60.0), total_duration_min)
            if h_end <= h_start:
                continue

            sub_steps = max(1, int(round((h_end - h_start) / max_dt_minutes)))
            base_sub_times = [
                round(h_start + i * ((h_end - h_start) / sub_steps), 6)
                for i in range(sub_steps + 1)
            ]
            for t in base_sub_times:
                get_s(t)

            intervals = [
                (base_sub_times[i], base_sub_times[i + 1])
                for i in range(len(base_sub_times) - 1)
            ]

            while intervals:
                ta, tb = intervals.pop()
                if (tb - ta) <= min_dt_minutes:
                    continue
                sa = get_s(ta)
                sb = get_s(tb)
                tm = round((ta + tb) / 2.0, 6)
                sm = get_s(tm)

                # 1. Error check at midpoint:
                err_1h = abs(sm.dead_1h - 0.5 * (sa.dead_1h + sb.dead_1h))
                err_10h = abs(sm.dead_10h - 0.5 * (sa.dead_10h + sb.dead_10h))

                # 2. Extinction crossing check:
                crosses_mx = False
                for mx in target_mx_list:
                    if (sa.dead_1h - mx) * (sb.dead_1h - mx) < 0:
                        crosses_mx = True
                        break
                    if abs(sm.dead_1h - mx) < 0.003 and (tb - ta) > (2.0 * min_dt_minutes):
                        crosses_mx = True
                        break

                if err_1h > fmc_error_tol or err_10h > fmc_error_tol or crosses_mx:
                    intervals.append((ta, tm))
                    intervals.append((tm, tb))

        sorted_times = sorted(list(states.keys()))
        d1_list = [states[t].dead_1h for t in sorted_times]
        d10_list = [states[t].dead_10h for t in sorted_times]
        d100_list = [states[t].dead_100h for t in sorted_times]

        times_arr = np.ascontiguousarray(sorted_times, dtype=np.float64)
        d1_arr = np.ascontiguousarray(d1_list, dtype=np.float64)
        d10_arr = np.ascontiguousarray(d10_list, dtype=np.float64)
        d100_arr = np.ascontiguousarray(d100_list, dtype=np.float64)

        times_arr.flags.writeable = False
        d1_arr.flags.writeable = False
        d10_arr.flags.writeable = False
        d100_arr.flags.writeable = False

        return NelsonFuelMoistureTimeSeries(
            times_minutes=times_arr,
            dead_1h=d1_arr,
            dead_10h=d10_arr,
            dead_100h=d100_arr,
            live_herbaceous=self._live_herbaceous,
            live_woody=self._live_woody,
            dt_minutes=max_dt_minutes,
            total_duration_minutes=total_duration_min,
            interpolation_method=interpolation_method,
            is_uniform=False,
        )


# ==============================================================================
# EXTINCTION GUARD EVALUATOR
# ==============================================================================

class ExtinctionGuardEvaluator:
    """
    Decoupled safety-critical evaluator for Rothermel moisture extinction proximity.

    Evaluates the exact physical variable controlling extinction in Q-NYX Rothermel:
        min_{k in dead} (mf_k - Mx_dead)

    When within guard_delta, caller can fall back to exact physical ODE integration
    (provider.get_state_exact(t)), eliminating extinction mismatches without slowing
    down the vast majority (>95%) of operational queries.
    """

    def __init__(self, guard_delta: Optional[float] = None, enabled: bool = False):
        self.guard_delta = guard_delta
        self.enabled = enabled

    @staticmethod
    def get_effective_dead_extinction_diff(state: FuelMoistureState, fuel_bed: Any) -> float:
        """
        Calculates min_{k in dead} (mf_k - Mx_dead).
        < 0: at least one dead particle class is combustible (spread possible).
        >= 0: all dead particle classes are at or above extinction (extinguished, ROS=0).
        """
        mx = float(fuel_bed.dead_extinction_moisture)
        diffs = []
        for p in fuel_bed.particle_classes:
            if not p.is_live:
                if p.name == "dead_1h":
                    diffs.append(state.dead_1h - mx)
                elif p.name == "dead_10h":
                    diffs.append(state.dead_10h - mx)
                elif p.name == "dead_100h":
                    diffs.append(state.dead_100h - mx)
        return min(diffs) if diffs else float("inf")

    def is_near_extinction(
        self,
        state: FuelMoistureState,
        fuel_bed: Any,
        delta: Optional[float] = None,
    ) -> bool:
        """
        Returns True if the approximate state is within delta of the physical extinction boundary.
        """
        d = delta if delta is not None else self.guard_delta
        if d is None or d <= 0.0:
            return False
        crit_diff = self.get_effective_dead_extinction_diff(state, fuel_bed)
        return abs(crit_diff) <= d

    def resolve_guarded_state(
        self,
        elapsed_minutes: float,
        provider: NelsonFuelMoistureProvider,
        fuel_bed: Any,
        delta: Optional[float] = None,
    ) -> Tuple[FuelMoistureState, bool]:
        """
        Resolves the fuel moisture state safely:
        1. Queries fast dense timeseries: s_fast = provider.get_state_fast(elapsed_minutes)
        2. Evaluates extinction proximity: is_near_extinction(s_fast, fuel_bed, delta)
        3. If near extinction: falls back to provider.get_state_exact(elapsed_minutes)
        4. Otherwise: returns s_fast.

        Returns (resolved_state, fallback_triggered).
        """
        s_approx = provider.get_state_fast(elapsed_minutes)
        if self.is_near_extinction(s_approx, fuel_bed, delta):
            s_exact = provider.get_state_exact(elapsed_minutes)
            return s_exact, True
        return s_approx, False


