"""
weather.py

Procesamiento de datos meteorológicos, soporte de series horarias discretas
(hourly_stepwise_weather) y vectores de viento (FROM / TO).
"""

import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from fire_engine.schemas import WeatherSeries


class WeatherError(ValueError):
    """Excepción base para errores del subsistema meteorológico."""
    pass


class WeatherRangeError(WeatherError):
    """Excepción cuando el tiempo de simulación solicitado excede el rango de datos disponibles."""
    pass


class MissingWeatherRecordError(WeatherError):
    """Excepción cuando falta un registro horario o existe discontinuidad temporal."""
    pass


def convert_wind_to_spread_heading(wind_direction_deg: float) -> float:
    """
    Convierte la dirección meteorológica del viento (dirección DESDE donde sopla, N=0°, E=90°)
    a la dirección de empuje/propagación del fuego (dirección HACIA donde sopla el viento).
    """
    return (float(wind_direction_deg) + 180.0) % 360.0


class WeatherManager:
    """
    Gestor de condiciones meteorológicas para simulación de propagación.
    Soporta dos modalidades:
    1. 'constant': Parámetros fijos para toda la duración de la corrida (backward compatible).
    2. 'hourly_stepwise': Serie horaria histórica por bloques [t, t+60 min).
    """
    def __init__(
        self,
        weather_data: Optional[WeatherSeries] = None,
        base_dir: Optional[Path] = None,
        hourly_records: Optional[List[Dict[str, Any]]] = None
    ):
        self.weather_initial = weather_data or WeatherSeries()
        self.status = self.weather_initial.status
        self.temporal_mode = getattr(self.weather_initial, "temporal_mode", "constant")
        self.start_datetime_utc = getattr(self.weather_initial, "start_datetime_utc", None)
        self.simulation_start_datetime_utc = getattr(self.weather_initial, "simulation_start_datetime_utc", None)
        self.hourly_records: List[Dict[str, Any]] = []
        self.sim_start_idx: int = 0

        # Validar radiación solar inicial si está definida
        if getattr(self.weather_initial, "solar_radiation_w_m2", None) is not None:
            sol = self.weather_initial.solar_radiation_w_m2
            if isinstance(sol, (int, float, np.floating, np.integer)):
                if np.isnan(sol) or np.isinf(sol):
                    raise WeatherError("solar_radiation_w_m2 contiene NaN o Inf.")
                if sol < 0.0:
                    raise WeatherError(f"solar_radiation_w_m2 no puede ser negativa ({sol} W/m²).")
            else:
                try:
                    sol_f = float(sol)
                    if np.isnan(sol_f) or np.isinf(sol_f):
                        raise WeatherError("solar_radiation_w_m2 contiene NaN o Inf.")
                    if sol_f < 0.0:
                        raise WeatherError(f"solar_radiation_w_m2 no puede ser negativa ({sol_f} W/m²).")
                except (ValueError, TypeError):
                    raise WeatherError(f"solar_radiation_w_m2 contiene valor no numérico '{sol}'.")

        if hourly_records is not None:
            self._set_hourly_records(hourly_records)
            self.temporal_mode = "hourly_stepwise"
        elif self.temporal_mode == "hourly_stepwise" and self.weather_initial.file is not None:
            file_path = self.weather_initial.file
            full_path = Path(file_path)
            if not full_path.is_absolute() and base_dir is not None:
                full_path = base_dir / full_path
            self.load_csv(full_path, start_datetime_utc=self.start_datetime_utc)
            self.temporal_mode = "hourly_stepwise"
        elif self.weather_initial.file is not None and str(self.weather_initial.file).lower().endswith(".csv"):
            file_path = self.weather_initial.file
            full_path = Path(file_path)
            if not full_path.is_absolute() and base_dir is not None:
                full_path = base_dir / full_path
            self.load_csv(full_path, start_datetime_utc=self.start_datetime_utc)
            self.temporal_mode = "hourly_stepwise"

    def _update_sim_start_idx(self):
        """Actualiza el índice correspondiente al inicio de la simulación."""
        if not self.hourly_records:
            self.sim_start_idx = 0
            return

        if self.simulation_start_datetime_utc is None:
            self.sim_start_idx = 0
            return

        target_dt = pd.to_datetime(self.simulation_start_datetime_utc, utc=True)
        found_idx = None
        for idx, rec in enumerate(self.hourly_records):
            dt_raw = rec.get("datetime_utc")
            if dt_raw is not None:
                try:
                    dt = pd.to_datetime(dt_raw, utc=True)
                    if dt == target_dt:
                        found_idx = idx
                        break
                except Exception:
                    pass

        if found_idx is None:
            first_dt = self.hourly_records[0].get("datetime_utc")
            last_dt = self.hourly_records[-1].get("datetime_utc")
            raise MissingWeatherRecordError(
                f"Timestamp de inicio de simulación '{self.simulation_start_datetime_utc}' "
                f"no encontrado en serie meteorológica. Rango disponible: {first_dt} a {last_dt}."
            )

        self.sim_start_idx = found_idx

    def is_time_dependent(self) -> bool:
        """Indica si el gestor meteorológico evoluciona en el tiempo."""
        return self.temporal_mode == "hourly_stepwise" and len(self.hourly_records) > 0

    def _set_hourly_records(self, records: List[Dict[str, Any]]):
        """Valida e indexa una lista de registros horarios en memoria."""
        if not records:
            raise MissingWeatherRecordError("La lista de registros meteorológicos horarios está vacía.")

        validated = []
        prev_dt = None

        for idx, r in enumerate(records):
            for req_field in ("wind_speed_ms", "wind_direction_deg"):
                if req_field not in r or r[req_field] is None or np.isnan(r[req_field]):
                    raise MissingWeatherRecordError(
                        f"Registro horario {idx} carece del campo obligatorio '{req_field}' o contiene NaN."
                    )

            ws = float(r["wind_speed_ms"])
            if ws < 0.0:
                raise WeatherError(f"Velocidad del viento negativa ({ws} m/s) en registro {idx}.")

            wdir = float(r["wind_direction_deg"]) % 360.0
            temp = float(r.get("temperature_c", self.weather_initial.temperature_c))
            rh = float(r.get("relative_humidity_pct", self.weather_initial.relative_humidity_pct))
            rain = float(r.get("precipitation_mm", 0.0))
            midflame = float(r["midflame_wind_ms"]) if "midflame_wind_ms" in r and r["midflame_wind_ms"] is not None else None
            dt_str = str(r.get("datetime_utc", f"hour_{idx}"))

            # Validar radiación solar si está presente
            sol_rad = r.get("solar_radiation_w_m2", getattr(self.weather_initial, "solar_radiation_w_m2", None))
            if sol_rad is not None:
                if isinstance(sol_rad, (int, float, np.floating, np.integer)):
                    if np.isnan(sol_rad) or np.isinf(sol_rad):
                        raise WeatherError(f"solar_radiation_w_m2 contiene NaN o Inf en registro {idx}.")
                    sol_val = float(sol_rad)
                else:
                    try:
                        sol_val = float(sol_rad)
                        if np.isnan(sol_val) or np.isinf(sol_val):
                            raise WeatherError(f"solar_radiation_w_m2 contiene NaN o Inf en registro {idx}.")
                    except (ValueError, TypeError):
                        raise WeatherError(f"solar_radiation_w_m2 con valor no numérico '{sol_rad}' en registro {idx}.")

                if sol_val < 0.0:
                    raise WeatherError(f"solar_radiation_w_m2 negativa ({sol_val} W/m²) en registro {idx}.")
                sol_rad = sol_val

            # Validar continuidad temporal si hay timestamps parseables
            if "datetime_utc" in r and r["datetime_utc"] is not None:
                try:
                    curr_dt = pd.to_datetime(r["datetime_utc"], utc=True)
                    if prev_dt is not None:
                        diff_hours = (curr_dt - prev_dt).total_seconds() / 3600.0
                        if not np.isclose(diff_hours, 1.0, atol=0.01):
                            raise MissingWeatherRecordError(
                                f"Discontinuidad temporal detectada en índice {idx}: salto de {diff_hours:.2f} h "
                                f"entre {prev_dt} y {curr_dt}. Se requiere paso horario estricto de 1.0 h."
                            )
                    prev_dt = curr_dt
                except Exception as e:
                    if isinstance(e, MissingWeatherRecordError):
                        raise
                    pass

            validated.append({
                "hour_index": idx,
                "datetime_utc": dt_str,
                "wind_speed_ms": ws,
                "wind_direction_deg": wdir,
                "temperature_c": temp,
                "relative_humidity_pct": rh,
                "precipitation_mm": rain,
                "solar_radiation_w_m2": sol_rad,
                "midflame_wind_ms": midflame
            })

        self.hourly_records = validated
        self._update_sim_start_idx()

    def load_csv(self, csv_path: Union[str, Path], start_datetime_utc: Optional[str] = None):
        """
        Carga y valida un archivo CSV con serie meteorológica horaria.
        """
        p = Path(csv_path)
        if not p.exists():
            raise FileNotFoundError(f"Archivo meteorológico no encontrado: {p}")

        df = pd.read_csv(p)

        # Mapeo flexible de nombres de columnas
        col_map = {}
        for c in df.columns:
            cl = c.strip().lower()
            if cl in ("datetime_utc", "valid_time", "time_utc", "timestamp"):
                col_map["datetime_utc"] = c
            elif cl in ("wind_speed_ms", "ws_ms", "wind_speed", "si10"):
                col_map["wind_speed_ms"] = c
            elif cl in ("wind_from_deg", "wind_direction_deg", "wind_dir_deg", "wdir_deg", "wdir"):
                col_map["wind_direction_deg"] = c
            elif cl in ("temperature_c", "temp_c", "t2m_c"):
                col_map["temperature_c"] = c
            elif cl in ("relative_humidity_pct", "rh_pct", "rh"):
                col_map["relative_humidity_pct"] = c
            elif cl in ("precipitation_mm", "precip_mm", "tp_mm"):
                col_map["precipitation_mm"] = c
            elif cl in ("midflame_wind_ms", "midflame_ms", "u_midflame"):
                col_map["midflame_wind_ms"] = c
            elif cl in ("solar_radiation_w_m2", "solar_radiation", "solar_rad_w_m2", "solar_rad", "ssrd_w_m2", "isolar"):
                col_map["solar_radiation_w_m2"] = c
            elif cl in ("ssrd", "ssrd_j_m2", "surface_solar_radiation_downwards"):
                col_map["ssrd_j_m2"] = c

        if "wind_speed_ms" not in col_map:
            raise MissingWeatherRecordError(f"Columna de velocidad de viento no encontrada en {p.name}. Columnas: {df.columns.tolist()}")
        if "wind_direction_deg" not in col_map:
            raise MissingWeatherRecordError(f"Columna de dirección de viento (FROM) no encontrada en {p.name}. Columnas: {df.columns.tolist()}")

        # Si hay datetime_utc, normalizar y filtrar por start_datetime_utc
        if "datetime_utc" in col_map:
            df["_dt"] = pd.to_datetime(df[col_map["datetime_utc"]], utc=True)
            df = df.sort_values("_dt").reset_index(drop=True)

            if start_datetime_utc is not None:
                start_dt = pd.to_datetime(start_datetime_utc, utc=True)
                match_mask = (df["_dt"] == start_dt)
                if not match_mask.any():
                    raise MissingWeatherRecordError(
                        f"Timestamp inicial '{start_datetime_utc}' no encontrado en serie {p.name}. "
                        f"Rango disponible: {df['_dt'].iloc[0]} a {df['_dt'].iloc[-1]}."
                    )
                start_idx = df.index[match_mask][0]
                df = df.iloc[start_idx:].reset_index(drop=True)

        records = []
        for idx, row in df.iterrows():
            # Ingestión de radiación solar
            solar_rad = None
            if "solar_radiation_w_m2" in col_map:
                val = row[col_map["solar_radiation_w_m2"]]
                if pd.notna(val):
                    val_f = float(val)
                    if np.isnan(val_f) or np.isinf(val_f):
                        raise WeatherError(f"solar_radiation_w_m2 contiene NaN o Inf en fila {idx}.")
                    if val_f < 0.0:
                        raise WeatherError(f"solar_radiation_w_m2 negativa ({val_f} W/m²) en fila {idx}.")
                    solar_rad = val_f
                else:
                    raise WeatherError(f"Valor NaN detectado en columna solar_radiation_w_m2 en fila {idx}.")
            elif "ssrd_j_m2" in col_map:
                raw_ssrd = row[col_map["ssrd_j_m2"]]
                if pd.notna(raw_ssrd):
                    raw_val = float(raw_ssrd)
                    if np.isnan(raw_val) or np.isinf(raw_val):
                        raise WeatherError(f"ssrd_j_m2 contiene NaN o Inf en fila {idx}.")
                    # Calcular duración real del intervalo temporal
                    if "_dt" in df.columns and len(df) > 1:
                        if idx == 0:
                            interval_sec = (df["_dt"].iloc[1] - df["_dt"].iloc[0]).total_seconds()
                        else:
                            interval_sec = (df["_dt"].iloc[idx] - df["_dt"].iloc[idx - 1]).total_seconds()
                        if interval_sec <= 0:
                            interval_sec = 3600.0
                    else:
                        interval_sec = 3600.0
                    solar_rad = max(0.0, raw_val / interval_sec)
                else:
                    raise WeatherError(f"Valor NaN detectado en columna ssrd_j_m2 en fila {idx}.")
            else:
                solar_rad = getattr(self.weather_initial, "solar_radiation_w_m2", None)

            rec = {
                "wind_speed_ms": row[col_map["wind_speed_ms"]],
                "wind_direction_deg": row[col_map["wind_direction_deg"]],
                "temperature_c": row[col_map["temperature_c"]] if "temperature_c" in col_map else self.weather_initial.temperature_c,
                "relative_humidity_pct": row[col_map["relative_humidity_pct"]] if "relative_humidity_pct" in col_map else self.weather_initial.relative_humidity_pct,
                "precipitation_mm": row[col_map["precipitation_mm"]] if "precipitation_mm" in col_map else 0.0,
                "solar_radiation_w_m2": solar_rad,
                "midflame_wind_ms": row[col_map["midflame_wind_ms"]] if "midflame_wind_ms" in col_map else None,
                "datetime_utc": str(row["_dt"]) if "_dt" in df.columns else str(row.get(col_map.get("datetime_utc"), f"hour_{idx}"))
            }
            records.append(rec)

        self._set_hourly_records(records)

    def get_simulation_records(self) -> List[Dict[str, Any]]:
        """Retorna los registros horarios a partir del inicio de la simulación t=0."""
        return self.hourly_records[self.sim_start_idx:]

    def get_conditioning_records(
        self,
        conditioning_hours: Optional[int] = None,
        conditioning_start_datetime_utc: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retorna la porción de registros meteorológicos previos al inicio de la simulación
        dedicada al acondicionamiento de humedad [-N, 0).
        """
        if conditioning_hours is None and conditioning_start_datetime_utc is None:
            return []

        if conditioning_start_datetime_utc is not None:
            target_dt = pd.to_datetime(conditioning_start_datetime_utc, utc=True)
            found_idx = None
            for idx, rec in enumerate(self.hourly_records):
                dt_raw = rec.get("datetime_utc")
                if dt_raw is not None:
                    try:
                        dt = pd.to_datetime(dt_raw, utc=True)
                        if dt == target_dt:
                            found_idx = idx
                            break
                    except Exception:
                        pass

            if found_idx is None:
                first_dt = self.hourly_records[0].get("datetime_utc") if self.hourly_records else "N/A"
                last_dt = self.hourly_records[-1].get("datetime_utc") if self.hourly_records else "N/A"
                raise MissingWeatherRecordError(
                    f"conditioning_start_datetime_utc '{conditioning_start_datetime_utc}' no encontrado "
                    f"en serie meteorológica. Rango disponible: {first_dt} a {last_dt}."
                )

            if found_idx > self.sim_start_idx:
                raise WeatherError(
                    f"conditioning_start_datetime_utc ({conditioning_start_datetime_utc}) es posterior al "
                    f"inicio de simulación ({self.simulation_start_datetime_utc or 'índice ' + str(self.sim_start_idx)})."
                )

            actual_hours = self.sim_start_idx - found_idx
            if conditioning_hours is not None and conditioning_hours != actual_hours:
                raise WeatherError(
                    f"Conflicto de configuración meteorológica: conditioning_hours={conditioning_hours} "
                    f"difiere del lapso entre conditioning_start_datetime_utc y simulation_start_datetime_utc ({actual_hours} horas)."
                )

            return self.hourly_records[found_idx:self.sim_start_idx]

        if conditioning_hours is not None:
            if conditioning_hours < 0:
                raise WeatherError(f"conditioning_hours no puede ser negativo: {conditioning_hours}.")
            if conditioning_hours == 0:
                return []

            start_idx = self.sim_start_idx - conditioning_hours
            if start_idx < 0:
                avail = self.sim_start_idx
                raise MissingWeatherRecordError(
                    f"Serie meteorológica insuficiente para conditioning: se solicitaron {conditioning_hours} horas, "
                    f"pero solo hay {avail} horas disponibles antes del inicio de la simulación."
                )

            return self.hourly_records[start_idx:self.sim_start_idx]

        return []

    def get_weather_at_time(self, elapsed_minutes: float) -> Dict[str, Any]:
        """
        Retorna los parámetros meteorológicos para un instante t (minutos).
        En modo 'hourly_stepwise', aplica selección por bloques [t, t+60 min).
        En modo 'constant', retorna los valores estáticos.
        """
        if elapsed_minutes < 0.0:
            raise WeatherRangeError(f"Tiempo de simulación negativo no válido: {elapsed_minutes} min.")

        if self.is_time_dependent():
            sim_h = int(math.floor(elapsed_minutes / 60.0))
            record_idx = self.sim_start_idx + sim_h

            if record_idx >= len(self.hourly_records):
                available_sim_hours = len(self.hourly_records) - self.sim_start_idx
                max_mins = available_sim_hours * 60.0
                raise WeatherRangeError(
                    f"Tiempo de simulación {elapsed_minutes:.2f} min (hora {sim_h}) excede el rango de "
                    f"datos horarios disponibles para simulación ({available_sim_hours} horas, máximo {max_mins:.1f} min)."
                )

            rec = self.hourly_records[record_idx]
            wind_speed = rec["wind_speed_ms"]
            wind_dir = rec["wind_direction_deg"]
            temp = rec["temperature_c"]
            rh = rec["relative_humidity_pct"]
            rain = rec["precipitation_mm"]
            sol_rad = rec.get("solar_radiation_w_m2")
            midflame = rec.get("midflame_wind_ms")
            dt_utc = rec.get("datetime_utc")
            spread_heading = convert_wind_to_spread_heading(wind_dir)

            return {
                "elapsed_minutes": elapsed_minutes,
                "hour_index": sim_h,
                "datetime_utc": dt_utc,
                "wind_speed_ms": wind_speed,
                "wind_direction_deg": wind_dir,
                "spread_heading_deg": spread_heading,
                "temperature_c": temp,
                "relative_humidity_pct": rh,
                "precipitation_mm": rain,
                "solar_radiation_w_m2": sol_rad,
                "midflame_wind_ms": midflame,
                "status": self.status,
                "temporal_mode": "hourly_stepwise"
            }

        # Modo constante (fallback idéntico para backward compatibility)
        wind_speed = self.weather_initial.wind_speed_ms
        wind_dir = self.weather_initial.wind_direction_deg
        temp = self.weather_initial.temperature_c
        rh = self.weather_initial.relative_humidity_pct
        rain = self.weather_initial.precipitation_mm
        sol_rad = getattr(self.weather_initial, "solar_radiation_w_m2", None)
        midflame = self.weather_initial.midflame_wind_ms
        spread_heading = convert_wind_to_spread_heading(wind_dir)

        return {
            "elapsed_minutes": elapsed_minutes,
            "hour_index": 0,
            "datetime_utc": self.simulation_start_datetime_utc or self.start_datetime_utc,
            "wind_speed_ms": wind_speed,
            "wind_direction_deg": wind_dir,
            "spread_heading_deg": spread_heading,
            "temperature_c": temp,
            "relative_humidity_pct": rh,
            "precipitation_mm": rain,
            "solar_radiation_w_m2": sol_rad,
            "midflame_wind_ms": midflame,
            "status": self.status,
            "temporal_mode": "constant"
        }
