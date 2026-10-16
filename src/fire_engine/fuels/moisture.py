"""
src/fire_engine/fuels/moisture.py

Arquitectura base de Fuel Moisture (Q-NYX FMC-1).
Define el contenedor inmutable FuelMoistureState, las interfaces de proveedores
y el proveedor constante (ConstantFuelMoistureProvider).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
import math
import numpy as np


class FuelMoistureError(ValueError):
    """Excepción base para errores del subsistema de humedad de combustible."""
    pass


class InvalidFuelMoistureValueError(FuelMoistureError):
    """Excepción cuando una fracción de humedad está fuera del rango físico admisible."""
    pass


class UnsupportedFuelMoistureModeError(FuelMoistureError, NotImplementedError):
    """Excepción cuando se solicita un modo de humedad no implementado en la etapa actual."""
    pass


# Rangos físicos válidos en fracción adimensional (mass_water / oven_dry_fuel_mass)
VALID_RANGES = {
    "dead_1h": (0.01, 0.60),
    "dead_10h": (0.01, 0.60),
    "dead_100h": (0.01, 0.60),
    "live_herbaceous": (0.30, 3.50),
    "live_woody": (0.50, 3.00),
}


# =============================================================================
# FMC-2.1: EQUILIBRIUM MOISTURE CONTENT (SIMARD 1968 / ROTHERMEL 1972)
# =============================================================================

def calculate_simard_emc(
    temperature_c: float,
    relative_humidity_pct: float
) -> float:
    """
    Calcula el Contenido de Humedad de Equilibrio (Equilibrium Moisture Content — EMC)
    para combustible fino muerto utilizando las correlaciones empíricas de Simard (1968)
    y Rothermel (1972, USDA Forest Service INT-115).

    Ecuación por tramos de Simard (1968):
        Para RH < 10%:
            EMC(%) = 0.03229 + 0.281073 * RH - 0.000578 * RH * T_F
        Para 10% <= RH <= 50%:
            EMC(%) = 2.22749 + 0.160107 * RH - 0.014784 * T_F
        Para RH > 50%:
            EMC(%) = 21.0606 + 0.005565 * (RH^2) - 0.00035 * RH * T_F - 0.483199 * RH

    Donde:
        - T_F es la temperatura del aire en grados Fahrenheit (°F): T_F = T_C * 1.8 + 32.0
        - RH es la humedad relativa del aire en porcentaje [0.0, 100.0]

    Retorna:
        moisture_ratio: Relación de masa adimensional (g_agua / g_seco),
                        donde por ejemplo 0.08 representa 8% de humedad.
    """
    if not isinstance(temperature_c, (int, float)) or math.isnan(temperature_c) or math.isinf(temperature_c):
        raise InvalidFuelMoistureValueError(
            f"Temperatura inválida: {temperature_c}. Debe ser un valor numérico finito."
        )

    if not isinstance(relative_humidity_pct, (int, float)) or math.isnan(relative_humidity_pct) or math.isinf(relative_humidity_pct):
        raise InvalidFuelMoistureValueError(
            f"Humedad relativa inválida: {relative_humidity_pct}. Debe ser un valor numérico finito."
        )

    rh = float(relative_humidity_pct)
    if rh < 0.0 or rh > 100.0:
        raise InvalidFuelMoistureValueError(
            f"Humedad relativa fuera de rango [0, 100]: {rh}%."
        )

    t_f = float(temperature_c) * 1.8 + 32.0

    if rh < 10.0:
        emc_pct = 0.03229 + 0.281073 * rh - 0.000578 * rh * t_f
    elif rh <= 50.0:
        emc_pct = 2.22749 + 0.160107 * rh - 0.014784 * t_f
    else:
        emc_pct = 21.0606 + 0.005565 * (rh ** 2) - 0.00035 * rh * t_f - 0.483199 * rh

    if emc_pct < 0.0:
        emc_pct = 0.0

    return float(emc_pct / 100.0)


# =============================================================================
# FMC-2.2: TIME-LAG ODE INTEGRATOR (BYRAM 1963 / FOSBERG 1970)
# =============================================================================

def integrate_timelag_step(
    m_initial: float,
    emc: float,
    timelag_hours: float,
    dt_hours: float = 1.0
) -> float:
    """
    Integra analíticamente un paso temporal de evolución de humedad de combustible
    muerto siguiendo la ecuación diferencial de retardo temporal (Byram 1963, Fosberg 1970):
        dM/dt = -(M - EMC) / tau
        M(t + dt) = EMC + (M(t) - EMC) * exp(-dt / tau)

    Parámetros:
        m_initial: Contenido de humedad inicial en relación de masa (g_agua / g_seco, >= 0).
        emc: Contenido de humedad de equilibrio en relación de masa (g_agua / g_seco, >= 0).
        timelag_hours: Tiempo de retardo característico tau en horas (> 0).
                       Ej: 1.0 para 1-h, 10.0 para 10-h, 100.0 para 100-h.
        dt_hours: Paso de tiempo de integración en horas (>= 0). Por defecto 1.0 h.

    Retorna:
        M(t + dt) como relación de masa (g_agua / g_seco).
    """
    if not isinstance(m_initial, (int, float)) or math.isnan(m_initial) or math.isinf(m_initial) or m_initial < 0.0:
        raise InvalidFuelMoistureValueError(
            f"m_initial inválido: {m_initial}. Debe ser un número finito >= 0."
        )

    if not isinstance(emc, (int, float)) or math.isnan(emc) or math.isinf(emc) or emc < 0.0:
        raise InvalidFuelMoistureValueError(
            f"emc inválido: {emc}. Debe ser un número finito >= 0."
        )

    if not isinstance(timelag_hours, (int, float)) or math.isnan(timelag_hours) or math.isinf(timelag_hours) or timelag_hours <= 0.0:
        raise InvalidFuelMoistureValueError(
            f"timelag_hours inválido: {timelag_hours}. Debe ser un número finito > 0."
        )

    if not isinstance(dt_hours, (int, float)) or math.isnan(dt_hours) or math.isinf(dt_hours) or dt_hours < 0.0:
        raise InvalidFuelMoistureValueError(
            f"dt_hours inválido: {dt_hours}. Debe ser un número finito >= 0."
        )

    m_init_f = float(m_initial)
    emc_f = float(emc)
    tau = float(timelag_hours)
    dt = float(dt_hours)

    if dt == 0.0:
        return m_init_f

    decay = math.exp(-dt / tau)
    m_next = emc_f + (m_init_f - emc_f) * decay

    return float(m_next)


@dataclass(frozen=True)
class FuelMoistureState:
    """
    Estado inmutable del contenido de humedad de combustible (FMC).
    Todas las unidades son relaciones de masa adimensionales:
        moisture_ratio = mass_water / oven_dry_fuel_mass (g_agua / g_seco)
        0.08 = 8 %
        1.20 = 120 %
    """
    dead_1h: float
    dead_10h: float
    dead_100h: float
    live_herbaceous: float
    live_woody: float

    def __post_init__(self):
        for field_name, (min_val, max_val) in VALID_RANGES.items():
            val = getattr(self, field_name)
            if val is None or math.isnan(val) or not (min_val <= val <= max_val):
                raise InvalidFuelMoistureValueError(
                    f"Valor inválido para '{field_name}': {val}. "
                    f"El rango físico admisible en fracción es [{min_val}, {max_val}]."
                )

    @classmethod
    def from_percentages(
        cls,
        dead_1h: float,
        dead_10h: float,
        dead_100h: float,
        live_herbaceous: float,
        live_woody: float
    ) -> "FuelMoistureState":
        """
        Constructor explícito a partir de valores en porcentaje (0% - 350%).
        Convierte inequívocamente % a fracción dividiendo por 100.0.
        """
        return cls(
            dead_1h=float(dead_1h) / 100.0,
            dead_10h=float(dead_10h) / 100.0,
            dead_100h=float(dead_100h) / 100.0,
            live_herbaceous=float(live_herbaceous) / 100.0,
            live_woody=float(live_woody) / 100.0
        )

    def as_dict(self) -> Dict[str, float]:
        return {
            "dead_1h": self.dead_1h,
            "dead_10h": self.dead_10h,
            "dead_100h": self.dead_100h,
            "live_herbaceous": self.live_herbaceous,
            "live_woody": self.live_woody,
        }

    def as_tuple(self) -> Tuple[float, float, float, float, float]:
        return (
            self.dead_1h,
            self.dead_10h,
            self.dead_100h,
            self.live_herbaceous,
            self.live_woody,
        )


class FuelMoistureProvider(ABC):
    """
    Interfaz abstracta del proveedor de humedad de combustible.
    Diseñada desde FMC-1 para soportar en el futuro consultas espacio-temporales
    fuel_moisture(x, y, t).
    """

    @abstractmethod
    def get_state(
        self,
        elapsed_minutes: float = 0.0,
        row: Optional[int] = None,
        col: Optional[int] = None
    ) -> FuelMoistureState:
        """
        Retorna el estado de humedad para un instante temporal y posición espacial dados.
        """
        pass

    def is_time_dependent(self) -> bool:
        """Indica si el proveedor evoluciona dinámicamente con el tiempo."""
        return False

    def is_spatially_variable(self) -> bool:
        """Indica si el proveedor entrega valores diferenciados por celda (row, col)."""
        return False


class ConstantFuelMoistureProvider(FuelMoistureProvider):
    """
    Proveedor de humedad de combustible constante para todo el dominio y tiempo.
    Devuelve exactamente el FuelMoistureState suministrado sin modificaciones ni redondeos.
    """

    def __init__(self, state: FuelMoistureState):
        if not isinstance(state, FuelMoistureState):
            raise TypeError(f"state debe ser una instancia de FuelMoistureState, obtenido {type(state)}")
        self._state = state

    @property
    def state(self) -> FuelMoistureState:
        return self._state

    def get_state(
        self,
        elapsed_minutes: float = 0.0,
        row: Optional[int] = None,
        col: Optional[int] = None
    ) -> FuelMoistureState:
        return self._state

    def is_time_dependent(self) -> bool:
        return False

    def is_spatially_variable(self) -> bool:
        return False


from fire_engine.weather import WeatherRangeError


@dataclass(frozen=True)
class ConditioningReport:
    """
    Reporte inmutable y auditable del proceso de acondicionamiento de humedad de combustible (Q-NYX FMC-5A).
    Registra trazabilidad completa de inputs meteorológicos, modo de semilla, evolución horaria,
    memoria residual matemática de la semilla y advertencias.
    """
    weather_data_start_datetime_utc: Optional[str]
    simulation_start_datetime_utc: Optional[str]
    conditioning_start_datetime_utc: Optional[str]
    conditioning_hours: int
    weather_records_used: int
    seed_mode: str
    seed_dead_1h: float
    seed_dead_10h: float
    seed_dead_100h: float
    final_dead_1h: float
    final_dead_10h: float
    final_dead_100h: float
    live_herbaceous: float
    live_woody: float
    dead_1h_seed_memory_fraction: float
    dead_10h_seed_memory_fraction: float
    dead_100h_seed_memory_fraction: float
    warnings: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "weather_data_start_datetime_utc": self.weather_data_start_datetime_utc,
            "simulation_start_datetime_utc": self.simulation_start_datetime_utc,
            "conditioning_start_datetime_utc": self.conditioning_start_datetime_utc,
            "conditioning_hours": self.conditioning_hours,
            "weather_records_used": self.weather_records_used,
            "seed_mode": self.seed_mode,
            "seed_dead_1h": self.seed_dead_1h,
            "seed_dead_10h": self.seed_dead_10h,
            "seed_dead_100h": self.seed_dead_100h,
            "final_dead_1h": self.final_dead_1h,
            "final_dead_10h": self.final_dead_10h,
            "final_dead_100h": self.final_dead_100h,
            "live_herbaceous": self.live_herbaceous,
            "live_woody": self.live_woody,
            "dead_1h_seed_memory_fraction": self.dead_1h_seed_memory_fraction,
            "dead_10h_seed_memory_fraction": self.dead_10h_seed_memory_fraction,
            "dead_100h_seed_memory_fraction": self.dead_100h_seed_memory_fraction,
            "warnings": list(self.warnings),
        }

    def to_dict(self) -> Dict[str, Any]:
        return self.as_dict()


# Umbral de advertencia ingenieril de memoria residual de la semilla.
# NOTA: 0.05 (5%) es una convención de diagnóstico / warning operacional, NO un límite físico.
# La magnitud física y matemática primaria es la función continua R_k = exp(-Delta_t / tau_k).
DEFAULT_RESIDUAL_MEMORY_WARNING_THRESHOLD: float = 0.05


def run_fuel_moisture_conditioning(
    conditioning_records: List[Dict[str, Any]],
    seed_mode: str = "equilibrium",
    seed_state: Optional[FuelMoistureState] = None,
    live_herbaceous: float = 0.60,
    live_woody: float = 0.90,
    weather_data_start_datetime_utc: Optional[str] = None,
    simulation_start_datetime_utc: Optional[str] = None,
    conditioning_start_datetime_utc: Optional[str] = None,
    residual_memory_warning_threshold: float = DEFAULT_RESIDUAL_MEMORY_WARNING_THRESHOLD,
) -> Tuple[FuelMoistureState, ConditioningReport]:
    """
    Ejecuta la integración temporal de acondicionamiento de combustible muerto (Q-NYX FMC-5A)
    sobre una serie meteorológica histórica continua [-N h, 0 h).
    """
    cond_hours = len(conditioning_records)

    # Determinar semilla
    if seed_mode == "equilibrium":
        if cond_hours == 0:
            raise FuelMoistureError(
                "seed_mode='equilibrium' requiere al menos un registro meteorológico para calcular el EMC semilla."
            )
        first_rec = conditioning_records[0]
        t_first = float(first_rec.get("temperature_c", 24.0))
        rh_first = float(first_rec.get("relative_humidity_pct", 38.0))
        emc_first = calculate_simard_emc(t_first, rh_first)
        s_d1 = emc_first
        s_d10 = emc_first
        s_d100 = emc_first
    elif seed_mode == "explicit":
        if seed_state is None:
            raise FuelMoistureError("seed_mode='explicit' requiere suministrar un seed_state (FuelMoistureState).")
        s_d1 = seed_state.dead_1h
        s_d10 = seed_state.dead_10h
        s_d100 = seed_state.dead_100h
    else:
        raise FuelMoistureError(f"seed_mode no reconocido: '{seed_mode}'. Permitidos: 'equilibrium', 'explicit'.")

    curr_d1 = s_d1
    curr_d10 = s_d10
    curr_d100 = s_d100

    for rec in conditioning_records:
        t = float(rec.get("temperature_c", 24.0))
        rh = float(rec.get("relative_humidity_pct", 38.0))
        emc = calculate_simard_emc(t, rh)
        curr_d1 = integrate_timelag_step(curr_d1, emc, timelag_hours=1.0, dt_hours=1.0)
        curr_d10 = integrate_timelag_step(curr_d10, emc, timelag_hours=10.0, dt_hours=1.0)
        curr_d100 = integrate_timelag_step(curr_d100, emc, timelag_hours=100.0, dt_hours=1.0)

    final_state = FuelMoistureState(
        dead_1h=curr_d1,
        dead_10h=curr_d10,
        dead_100h=curr_d100,
        live_herbaceous=live_herbaceous,
        live_woody=live_woody
    )

    mem_1h = math.exp(-cond_hours / 1.0)
    mem_10h = math.exp(-cond_hours / 10.0)
    mem_100h = math.exp(-cond_hours / 100.0)

    warnings = []
    if mem_10h > residual_memory_warning_threshold:
        warnings.append(
            f"CONDITIONING_RESIDUAL_MEMORY_10H: {cond_hours}h conditioning retains {mem_10h*100.0:.2f}% "
            f"of initial 10-h seed."
        )
    if mem_100h > residual_memory_warning_threshold:
        warnings.append(
            f"CONDITIONING_RESIDUAL_MEMORY_100H: {cond_hours}h conditioning retains {mem_100h*100.0:.2f}% "
            f"of initial 100-h seed."
        )

    report = ConditioningReport(
        weather_data_start_datetime_utc=weather_data_start_datetime_utc,
        simulation_start_datetime_utc=simulation_start_datetime_utc,
        conditioning_start_datetime_utc=conditioning_start_datetime_utc or (conditioning_records[0].get("datetime_utc") if conditioning_records else None),
        conditioning_hours=cond_hours,
        weather_records_used=cond_hours,
        seed_mode=seed_mode,
        seed_dead_1h=s_d1,
        seed_dead_10h=s_d10,
        seed_dead_100h=s_d100,
        final_dead_1h=curr_d1,
        final_dead_10h=curr_d10,
        final_dead_100h=curr_d100,
        live_herbaceous=live_herbaceous,
        live_woody=live_woody,
        dead_1h_seed_memory_fraction=mem_1h,
        dead_10h_seed_memory_fraction=mem_10h,
        dead_100h_seed_memory_fraction=mem_100h,
        warnings=warnings
    )

    return final_state, report


class DynamicHourlyFuelMoistureProvider(FuelMoistureProvider):
    """
    Proveedor dinámico de humedad de combustible para combustibles muertos (Q-NYX FMC-2B).
    Evoluciona las fracciones de dead_1h, dead_10h y dead_100h en función de las series
    horarias de temperatura y humedad relativa (Simard EMC + time-lag ODE).
    Mantiene constantes las fracciones vivas (live_herbaceous, live_woody).

    Es determinista e independiente del orden de consulta temporal get_state(t).
    """

    def __init__(
        self,
        initial_state: FuelMoistureState,
        weather_manager: Any = None,
        hourly_records: Optional[List[Dict[str, Any]]] = None
    ):
        if not isinstance(initial_state, FuelMoistureState):
            raise TypeError(f"initial_state debe ser FuelMoistureState, obtenido {type(initial_state)}")
        self._initial_state = initial_state
        self._weather_manager = weather_manager
        self.conditioning_report: Optional[ConditioningReport] = None

        if hourly_records is not None:
            self._records = hourly_records
        elif weather_manager is not None and hasattr(weather_manager, "get_simulation_records"):
            self._records = weather_manager.get_simulation_records()
        elif weather_manager is not None and hasattr(weather_manager, "hourly_records"):
            self._records = weather_manager.hourly_records
        else:
            raise FuelMoistureError(
                "DynamicHourlyFuelMoistureProvider requiere un WeatherManager con registros horarios "
                "o una lista explícita de hourly_records."
            )

        if not self._records:
            raise FuelMoistureError("La lista de registros horarios de meteorología está vacía.")

        self._num_hours = len(self._records)
        self._precompute_hourly_states()

    def _precompute_hourly_states(self):
        """
        Precomputa determinísticamente los estados al inicio de cada hora entera:
        _hourly_states[0] = initial_state (en t = 0 min)
        _hourly_states[h+1] = evolución de _hourly_states[h] tras 1 hora con weather[h]
        """
        states = [self._initial_state]
        curr_state = self._initial_state

        for h, rec in enumerate(self._records):
            temp_c = float(rec.get("temperature_c", 24.0))
            rh_pct = float(rec.get("relative_humidity_pct", 38.0))
            emc = calculate_simard_emc(temp_c, rh_pct)

            next_d1h = integrate_timelag_step(
                m_initial=curr_state.dead_1h,
                emc=emc,
                timelag_hours=1.0,
                dt_hours=1.0
            )
            next_d10h = integrate_timelag_step(
                m_initial=curr_state.dead_10h,
                emc=emc,
                timelag_hours=10.0,
                dt_hours=1.0
            )
            next_d100h = integrate_timelag_step(
                m_initial=curr_state.dead_100h,
                emc=emc,
                timelag_hours=100.0,
                dt_hours=1.0
            )

            next_state = FuelMoistureState(
                dead_1h=next_d1h,
                dead_10h=next_d10h,
                dead_100h=next_d100h,
                live_herbaceous=self._initial_state.live_herbaceous,
                live_woody=self._initial_state.live_woody
            )
            states.append(next_state)
            curr_state = next_state

        self._hourly_states = states

    @property
    def initial_state(self) -> FuelMoistureState:
        return self._initial_state

    @property
    def hourly_states(self) -> List[FuelMoistureState]:
        return self._hourly_states

    def is_time_dependent(self) -> bool:
        return True

    def is_spatially_variable(self) -> bool:
        return False

    def get_state(
        self,
        elapsed_minutes: float = 0.0,
        row: Optional[int] = None,
        col: Optional[int] = None
    ) -> FuelMoistureState:
        """
        Retorna determinísticamente el FuelMoistureState para cualquier elapsed_minutes.
        Si elapsed_minutes cae dentro de una hora [h*60, (h+1)*60), integra desde el estado
        _hourly_states[h] usando la meteorología de la hora h y dt_hours = (elapsed_minutes - h*60)/60.
        """
        if elapsed_minutes < 0.0:
            raise WeatherRangeError(f"Tiempo de simulación negativo no válido: {elapsed_minutes} min.")

        h = int(math.floor(elapsed_minutes / 60.0))

        if h >= self._num_hours:
            max_mins = self._num_hours * 60.0
            raise WeatherRangeError(
                f"Tiempo de simulación {elapsed_minutes:.2f} min (hora {h}) excede el rango de "
                f"datos meteorológicos disponibles ({self._num_hours} horas, máximo {max_mins:.1f} min)."
            )

        remainder_min = elapsed_minutes - (h * 60.0)

        # Si coincide exactamente con el inicio de la hora entera
        if math.isclose(remainder_min, 0.0, abs_tol=1e-9):
            return self._hourly_states[h]

        # Evolución continua intra-horaria
        base_state = self._hourly_states[h]
        rec = self._records[h]
        temp_c = float(rec.get("temperature_c", 24.0))
        rh_pct = float(rec.get("relative_humidity_pct", 38.0))
        emc = calculate_simard_emc(temp_c, rh_pct)
        dt_hours = remainder_min / 60.0

        d1 = integrate_timelag_step(base_state.dead_1h, emc, 1.0, dt_hours)
        d10 = integrate_timelag_step(base_state.dead_10h, emc, 10.0, dt_hours)
        d100 = integrate_timelag_step(base_state.dead_100h, emc, 100.0, dt_hours)

        return FuelMoistureState(
            dead_1h=d1,
            dead_10h=d10,
            dead_100h=d100,
            live_herbaceous=self._initial_state.live_herbaceous,
            live_woody=self._initial_state.live_woody
        )


# Alias de compatibilidad
DynamicFuelMoistureProvider = DynamicHourlyFuelMoistureProvider


class ExternalFuelMoistureProvider(FuelMoistureProvider):
    """Placeholder formal para FMC-6 (tensores 2D/3D externos)."""
    def __init__(self, *args, **kwargs):
        raise UnsupportedFuelMoistureModeError(
            "El modo 'external' no está implementado en la etapa FMC-2B. "
            "Está planificado para la etapa FMC-6."
        )

    def get_state(self, elapsed_minutes: float = 0.0, row: Optional[int] = None, col: Optional[int] = None) -> FuelMoistureState:
        raise UnsupportedFuelMoistureModeError("external no implementado en FMC-2B.")


def create_fuel_moisture_provider(
    fuel_moisture_config: Optional[Any] = None,
    legacy_initial_moisture: Optional[Any] = None,
    weather_manager: Optional[Any] = None
) -> FuelMoistureProvider:
    """
    Factory function para instanciar el FuelMoistureProvider correspondiente
    a partir de la configuración del escenario o del bloque legacy initial_fuel_moisture.
    """
    # 1. Si existe configuración nueva de fuel_moisture
    if fuel_moisture_config is not None:
        mode = getattr(fuel_moisture_config, "mode", "constant")
        init_mode = getattr(fuel_moisture_config, "initialization_mode", "explicit")
        physics_model = getattr(fuel_moisture_config, "physics_model", "simple_timelag") or "simple_timelag"
        physics_model = physics_model.lower().strip()

        lh = getattr(fuel_moisture_config, "live_herbaceous_fraction", None) or getattr(fuel_moisture_config, "live_herbaceous", None)
        lw = getattr(fuel_moisture_config, "live_woody_fraction", None) or getattr(fuel_moisture_config, "live_woody", None)

        if legacy_initial_moisture is not None:
            lh = lh if lh is not None else getattr(legacy_initial_moisture, "live_herbaceous_fraction", 0.60)
            lw = lw if lw is not None else getattr(legacy_initial_moisture, "live_woody_fraction", 0.90)
        else:
            lh = lh if lh is not None else 0.60
            lw = lw if lw is not None else 0.90

        lh = float(lh)
        lw = float(lw)

        # Ruta Opt-In: Modelo Físico Nelson (Q-NYX NELSON-003A)
        if physics_model == "nelson":
            if mode != "hourly_dynamic":
                raise FuelMoistureError(
                    f"physics_model='nelson' requiere mode='hourly_dynamic', pero se recibió mode='{mode}'."
                )
            if weather_manager is None:
                raise FuelMoistureError(
                    "El modelo Nelson requiere suministrar un weather_manager con serie horaria."
                )

            from fire_engine.schemas import VALID_NELSON_RUNTIME_MODES
            from fire_engine.fuels.nelson import NelsonFuelMoistureProvider

            runtime_mode = getattr(fuel_moisture_config, "runtime_mode", None) or "hourly_anchored_exact_sampled_5min"
            runtime_mode = runtime_mode.lower().strip()
            if runtime_mode not in VALID_NELSON_RUNTIME_MODES:
                raise FuelMoistureError(
                    f"Unknown runtime_mode '{runtime_mode}' para Nelson. "
                    f"Modos válidos permitidos: {sorted(list(VALID_NELSON_RUNTIME_MODES))}"
                )

            d1 = getattr(fuel_moisture_config, "dead_1h_fraction", None) or getattr(fuel_moisture_config, "dead_1h", None)
            d10 = getattr(fuel_moisture_config, "dead_10h_fraction", None) or getattr(fuel_moisture_config, "dead_10h", None)
            d100 = getattr(fuel_moisture_config, "dead_100h_fraction", None) or getattr(fuel_moisture_config, "dead_100h", None)

            if legacy_initial_moisture is not None:
                d1 = d1 if d1 is not None else getattr(legacy_initial_moisture, "dead_1h_fraction", 0.20)
                d10 = d10 if d10 is not None else getattr(legacy_initial_moisture, "dead_10h_fraction", 0.20)
                d100 = d100 if d100 is not None else getattr(legacy_initial_moisture, "dead_100h_fraction", 0.20)
            else:
                d1 = d1 if d1 is not None else 0.20
                d10 = d10 if d10 is not None else 0.20
                d100 = d100 if d100 is not None else 0.20

            if init_mode == "conditioned":
                cond_hours = getattr(fuel_moisture_config, "conditioning_hours", None)
                cond_start_dt = getattr(fuel_moisture_config, "conditioning_start_datetime_utc", None)
                seed_mode = getattr(fuel_moisture_config, "seed_mode", "nfdrs_reference")
                cond_records = weather_manager.get_conditioning_records(
                    conditioning_hours=cond_hours,
                    conditioning_start_datetime_utc=cond_start_dt
                )
                return NelsonFuelMoistureProvider(
                    weather_manager=weather_manager,
                    conditioning_records=cond_records,
                    initialization_mode="conditioned",
                    seed_mode=seed_mode,
                    live_herbaceous=lh,
                    live_woody=lw,
                    seed_dead_1h=float(d1),
                    seed_dead_10h=float(d10),
                    seed_dead_100h=float(d100),
                    conditioning_hours=cond_hours,
                    conditioning_start_datetime_utc=cond_start_dt,
                    lookup_mode=runtime_mode,
                )
            elif init_mode == "explicit":
                return NelsonFuelMoistureProvider(
                    weather_manager=weather_manager,
                    initialization_mode="explicit",
                    seed_mode="explicit",
                    live_herbaceous=lh,
                    live_woody=lw,
                    seed_dead_1h=float(d1),
                    seed_dead_10h=float(d10),
                    seed_dead_100h=float(d100),
                    lookup_mode=runtime_mode,
                )
            else:
                raise UnsupportedFuelMoistureModeError(
                    f"Modo de inicialización no reconocido: '{init_mode}'. Permitidos: 'explicit', 'conditioned'."
                )

        if init_mode == "conditioned":
            if weather_manager is None:
                raise FuelMoistureError(
                    "El modo de inicialización 'conditioned' requiere suministrar un weather_manager con serie meteorológica."
                )

            cond_hours = getattr(fuel_moisture_config, "conditioning_hours", None)
            cond_start_dt = getattr(fuel_moisture_config, "conditioning_start_datetime_utc", None)
            seed_mode = getattr(fuel_moisture_config, "seed_mode", "equilibrium")

            cond_records = weather_manager.get_conditioning_records(
                conditioning_hours=cond_hours,
                conditioning_start_datetime_utc=cond_start_dt
            )

            seed_state = None
            if seed_mode == "explicit":
                d1 = getattr(fuel_moisture_config, "dead_1h_fraction", None) or getattr(fuel_moisture_config, "dead_1h", None)
                d10 = getattr(fuel_moisture_config, "dead_10h_fraction", None) or getattr(fuel_moisture_config, "dead_10h", None)
                d100 = getattr(fuel_moisture_config, "dead_100h_fraction", None) or getattr(fuel_moisture_config, "dead_100h", None)

                if legacy_initial_moisture is not None:
                    d1 = d1 if d1 is not None else getattr(legacy_initial_moisture, "dead_1h_fraction", 0.08)
                    d10 = d10 if d10 is not None else getattr(legacy_initial_moisture, "dead_10h_fraction", 0.10)
                    d100 = d100 if d100 is not None else getattr(legacy_initial_moisture, "dead_100h_fraction", 0.12)
                else:
                    d1 = d1 if d1 is not None else 0.08
                    d10 = d10 if d10 is not None else 0.10
                    d100 = d100 if d100 is not None else 0.12

                seed_state = FuelMoistureState(
                    dead_1h=float(d1),
                    dead_10h=float(d10),
                    dead_100h=float(d100),
                    live_herbaceous=lh,
                    live_woody=lw
                )

            weather_data_start = getattr(weather_manager, "start_datetime_utc", None)
            sim_start = getattr(weather_manager, "simulation_start_datetime_utc", None)

            conditioned_state, report = run_fuel_moisture_conditioning(
                conditioning_records=cond_records,
                seed_mode=seed_mode,
                seed_state=seed_state,
                live_herbaceous=lh,
                live_woody=lw,
                weather_data_start_datetime_utc=weather_data_start,
                simulation_start_datetime_utc=sim_start,
                conditioning_start_datetime_utc=cond_start_dt
            )

            if mode == "constant":
                provider = ConstantFuelMoistureProvider(conditioned_state)
                setattr(provider, "conditioning_report", report)
                return provider
            elif mode == "hourly_dynamic":
                provider = DynamicHourlyFuelMoistureProvider(
                    initial_state=conditioned_state,
                    weather_manager=weather_manager
                )
                setattr(provider, "conditioning_report", report)
                return provider
            elif mode == "external":
                raise UnsupportedFuelMoistureModeError(
                    "El modo 'external' no está implementado en la etapa FMC-5A. "
                    "Está planificado para la etapa FMC-6."
                )
            else:
                raise UnsupportedFuelMoistureModeError(f"Modo de fuel_moisture no reconocido: '{mode}'.")

        elif init_mode != "explicit":
            raise UnsupportedFuelMoistureModeError(
                f"Modo de inicialización no reconocido: '{init_mode}'. Permitidos: 'explicit', 'conditioned'."
            )

        # Extraer valores para el estado inicial explícito
        d1 = getattr(fuel_moisture_config, "dead_1h_fraction", None) or getattr(fuel_moisture_config, "dead_1h", None)
        d10 = getattr(fuel_moisture_config, "dead_10h_fraction", None) or getattr(fuel_moisture_config, "dead_10h", None)
        d100 = getattr(fuel_moisture_config, "dead_100h_fraction", None) or getattr(fuel_moisture_config, "dead_100h", None)

        if legacy_initial_moisture is not None:
            d1 = d1 if d1 is not None else getattr(legacy_initial_moisture, "dead_1h_fraction", 0.08)
            d10 = d10 if d10 is not None else getattr(legacy_initial_moisture, "dead_10h_fraction", 0.10)
            d100 = d100 if d100 is not None else getattr(legacy_initial_moisture, "dead_100h_fraction", 0.12)
        else:
            d1 = d1 if d1 is not None else 0.08
            d10 = d10 if d10 is not None else 0.10
            d100 = d100 if d100 is not None else 0.12

        state = FuelMoistureState(
            dead_1h=float(d1),
            dead_10h=float(d10),
            dead_100h=float(d100),
            live_herbaceous=lh,
            live_woody=lw
        )

        if mode == "constant":
            return ConstantFuelMoistureProvider(state)
        elif mode == "hourly_dynamic":
            if weather_manager is None:
                raise FuelMoistureError(
                    "El modo 'hourly_dynamic' requiere suministrar un weather_manager con serie horaria."
                )
            return DynamicHourlyFuelMoistureProvider(
                initial_state=state,
                weather_manager=weather_manager
            )
        elif mode == "external":
            raise UnsupportedFuelMoistureModeError(
                "El modo 'external' no está implementado en la etapa FMC-2B. "
                "Está planificado para la etapa FMC-6."
            )
        else:
            raise UnsupportedFuelMoistureModeError(f"Modo de fuel_moisture no reconocido: '{mode}'.")

    # 2. Ruta Legacy: initial_fuel_moisture
    if legacy_initial_moisture is not None:
        d1 = getattr(legacy_initial_moisture, "dead_1h_fraction", 0.08)
        d10 = getattr(legacy_initial_moisture, "dead_10h_fraction", 0.10)
        d100 = getattr(legacy_initial_moisture, "dead_100h_fraction", 0.12)
        lh = getattr(legacy_initial_moisture, "live_herbaceous_fraction", 0.60)
        lw = getattr(legacy_initial_moisture, "live_woody_fraction", None)
        lw_val = float(lw) if lw is not None else 0.90  # Default seguro para vector de 5 clases

        state = FuelMoistureState(
            dead_1h=float(d1),
            dead_10h=float(d10),
            dead_100h=float(d100),
            live_herbaceous=float(lh),
            live_woody=lw_val
        )
        return ConstantFuelMoistureProvider(state)

    # 3. Default fallback estándar
    default_state = FuelMoistureState(
        dead_1h=0.08,
        dead_10h=0.10,
        dead_100h=0.12,
        live_herbaceous=0.60,
        live_woody=0.90
    )
    return ConstantFuelMoistureProvider(default_state)
