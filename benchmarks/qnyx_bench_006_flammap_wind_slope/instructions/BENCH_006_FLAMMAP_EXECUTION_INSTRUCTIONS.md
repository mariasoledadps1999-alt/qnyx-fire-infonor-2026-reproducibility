# INSTRUCCIONES DE EJECUCIÓN MANUAL DE FLAMMAP 6.2 — BENCHMARK QNYX-BENCH-006

**Benchmark:** QNYX-BENCH-006 (Controlled Q-NYX vs FlamMap Wind-Slope Interaction Comparison)
**Versión Esperada:** FlamMap 6.2.45.0
**Modo de Simulación:** Basic Fire Behavior + MTT (Minimum Travel Time)
**Horizonte Temporal:** 180.0 minutos
**Estado en Fase 5.4B:** LCP BUILT & READY FOR EXECUTION — The six official Landscape files (.LCP) have been built manually in FlamMap GUI, audited and frozen. The execution package is fully prepared for Phase 5.4C.

---

## 1. Archivos de Paisaje (.LCP) Oficiales Congelados

Los seis archivos `.lcp` se encuentran en `flammap/landscapes/`:

| Landscape ID | Archivo LCP | Topografía Fuente | Casos Asociados | Reopen Gate | Estado |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LCP01_flat** | `bench006_flat.lcp` | `topo_flat_0deg` | Case 01, Case 02 | `PASS` (Reabierto) | `BUILT` |
| **LCP02_slope10_up_east** | `bench006_slope10_up_east.lcp` | `topo_slope_10deg_upslope_east` | Case 03, Case 04 | `NOT_INDIVIDUALLY_TESTED` | `BUILT` |
| **LCP03_slope10_up_ne45** | `bench006_slope10_up_ne45.lcp` | `topo_slope_10deg_upslope_ne45` | Case 05 | `NOT_INDIVIDUALLY_TESTED` | `BUILT` |
| **LCP04_slope10_up_north** | `bench006_slope10_up_north.lcp` | `topo_slope_10deg_upslope_north` | Case 06 | `NOT_INDIVIDUALLY_TESTED` | `BUILT` |
| **LCP05_slope10_up_sw225** | `bench006_slope10_up_sw225.lcp` | `topo_slope_10deg_upslope_sw225` | Case 07 | `NOT_INDIVIDUALLY_TESTED` | `BUILT` |
| **LCP06_slope10_up_west** | `bench006_slope10_up_west.lcp` | `topo_slope_10deg_upslope_west` | Case 08 | `NOT_INDIVIDUALLY_TESTED` | `BUILT` |

Propiedades comunes de los 6 LCPs:
- CRS: `EPSG:32719`
- Dimensiones: `301 rows x 301 cols`, resolución `30.0 m`
- Lower Left: $X = 188850.0\text{ m}, Y = 6068550.0\text{ m}$
- Crown fuel themes: Capas neutrales (0 cover, 0 height, 0 base height, 0 bulk density).
- Modalidad: **SURFACE FIRE ONLY**.

---

## 2. Configuración Ambiental y Meteorológica en FlamMap

1. **Archivo de Humedad (.FMS):**
   - Cargar directamente `flammap/common/flame_benchmark.fms`.
   - Contenido exacto congelado: `1 8 9 10 100 100 10`
   - Foliar Moisture: 100%
   - No editar valores manualmente en la interfaz.

2. **Viento:**
   - Casos sin viento (01 y 03): `Wind Speed = 0.0 mph`, `Wind Direction = 0°`.
   - Casos con viento (02, 04, 05, 06, 07, 08): `Wind Speed = 5.0 mph` (U20 a 20 ft), `Wind Direction = 270.0°` (azimut FROM West -> propaga al Este 90°).

3. **Cálculo de Copas:**
   - Mantener `Include Crown Fuels = OFF` (Fuego superficial únicamente).

---

## 3. Configuración de Ignición y MTT

1. **Ignición:**
   - Cargar shapefile de punto: `flammap/common/ignition_center.shp` (Punto único en $X = 193365.0\text{ m}, Y = 6073065.0\text{ m}$, EPSG:32719).
   - Corresponde exactamente a la celda central (Fila 150, Columna 150).
2. **Parámetros MTT:**
   - Simulación Duration: `180.0 min`.
   - Calculation Resolution: `30.0 m`.
   - Path Interval: `500.0 m`.
   - Spotting: Probabilidad = 0.0 (desactivado).
   - Búsqueda lateral: 6, búsqueda vertical: 4.

---

## 4. Exportación de Capas Requeridas en Fase 5.4C

Exportar a `flammap/<case_id>/results/`:
- `flammap_basic_ros.tif` (Basic Rate of Spread en m/min)
- `flammap_max_spread_direction.tif` (Maximum Spread Direction en grados)
- `flammap_midflame_wind.tif` (Midflame Wind Speed en m/s)
- `flammap_mtt_arrival_time.tif` (MTT Arrival Time en minutos)
- `flammap_mtt_ros.tif` (MTT Rate of Spread en m/min)

> [!IMPORTANT]
> Si la interfaz de FlamMap muestra nombres de botones o formatos ligeramente diferentes, marcar en el registro `VERIFY_IN_INSTALLED_FLAMMAP_UI` y aplicar la conversión estándar a unidades SI ($1\text{ ft/min} = 0.3048\text{ m/min}$) antes de guardar el GeoTIFF final.
