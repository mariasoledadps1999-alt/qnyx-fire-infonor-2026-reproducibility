# Data Management, Provenance & Licensing Guide

This document details the external datasets and inputs required to reproduce the experiments presented in the INFONOR 2026 paper:
**"Q-NYX Fire: Deterministic Wildfire Propagation Engine, Historical Validation, and Controlled Benchmark Against FlamMap"**.

---

## 1. External Data Sources Overview

| Dataset Identifier | Primary Source / Provider | License / Terms | Inclusion in Repository | Verification Method |
| :--- | :--- | :--- | :--- | :--- |
| **NASADEM Topography** | NASA JPL / USGS (SRTM mission recalibration) | Public Domain (US Gov Work) | Included as cropped 30m GeoTIFFs (`b0/input/`) | SHA-256 Checksum |
| **ERA5-Land Meteorology** | Copernicus Climate Change Service (C3S) / ECMWF | CC-BY 4.0 / Copernicus Open Access | Included as canonical processed NetCDF & CSVs | SHA-256 Checksum |
| **CONAF Reference Perimeter** | Corporación Nacional Forestal (CONAF), Chile | Datos Abiertos del Estado de Chile (Ley 20.285) | Included as vector GeoPackage (`conaf_reference/`) | SHA-256 Checksum |
| **Scott & Burgan Fuel Catalog** | USDA Forest Service (RMRS-GTR-153) | Public Domain (US Federal Government) | Included as structured JSON catalog (`configs/`) | SHA-256 Checksum |
| **FlamMap Benchmark Matrices** | USDA Forest Service / Missoula Fire Lab | Public Domain / Open Benchmark | Included as synthetic rasters & LCPs (`qnyx_bench_006_flammap_wind_slope/`) | SHA-256 Checksum |

---

## 2. Dataset Specifications & Acquisition

### 2.1 NASADEM Topography (30m Resolution)
- **Source Tile**: `NASADEM_HGT_s34w071`
- **Official Portal**: [NASA Earthdata Search](https://search.earthdata.nasa.gov/) or [Microsoft Planetary Computer STAC API](https://planetarycomputer.microsoft.com/dataset/nasadem).
- **Coordinate Reference System**: `EPSG:32719` (WGS 84 / UTM zone 19S).
- **Domain Extent**:
  - `xmin`: 299,760.0 m E
  - `ymin`: 6,217,530.0 m N
  - `xmax`: 329,220.0 m E
  - `ymax`: 6,236,730.0 m N
  - Dimensions: 640 rows × 982 columns (56,563.20 ha).
- **Layers Derived**:
  - `elevation.tif`: NASADEM digital elevation model [meters above sea level].
  - `slope.tif`: Horn (1981) terrain gradient [degrees].
  - `aspect_qnyx.tif`: Downslope exposition compass angle [0°=North, 90°=East].
  - `barriers.tif`: Water bodies, paved roads, and urban barriers [0=Burnable, 1=Barrier].

### 2.2 ERA5-Land Reanalysis Meteorology
- **Provider**: Copernicus Climate Change Service (C3S) / ECMWF.
- **Access Protocol**: Climate Data Store (CDS) API (`cdsapi`).
- **Variables Queried**:
  - `10m_u_component_of_wind` ($u_{10}$, m/s)
  - `10m_v_component_of_wind` ($v_{10}$, m/s)
  - `2m_temperature` ($T_{2\text{m}}$, K)
  - `2m_dewpoint_temperature` ($T_{d,2\text{m}}$, K)
  - `surface_solar_radiation_downwards` ($R_{\text{sol}}$, J/m²)
  - `total_precipitation` ($P$, m)
- **Temporal Window**: `2017-01-12 00:00:00 UTC` to `2017-01-16 23:00:00 UTC`.
- **Wind Reduction**: Midflame wind adjustment factor $\text{WAF} = 0.40$ applied to 10m wind speed ($U_{\text{mid}} = 0.40 \cdot U_{10\text{m}}$).
- **Mandatory Attribution**:
  > *"Generated using Copernicus Climate Change Service information [2017]. Neither the European Commission nor ECMWF is responsible for any use that may be made of the information."*

### 2.3 CONAF Historical Perimeter Database (Chile)
- **Entity**: Corporación Nacional Forestal (CONAF), Ministerio de Agricultura de Chile.
- **Incident**: Incendio Forestal N° 261 `"TALAMI"` (Temporada 2016–2017).
- **Dates**:
  - Ignition Date Reported: `2017-01-14`
  - Extinction / Control Date: `2017-02-09` (26 days active duration).
- **Area Values**:
  - Declared Administrative Area: **7,606.51 ha**
  - Geometric Vector Area (EPSG:32719): **8,002.90 ha**
  - Rasterized Area (30m pixel center): **8,006.22 ha** (88,958 pixels)
- **Role in Research**:
  - Evaluated strictly as a **Final Spatial Containment Envelope Reference**.
  - **Scientific Guard**: It is scientifically invalid to report a "12-hour simulation error vs CONAF final perimeter", because CONAF records the cumulative 26-day perimeter, whereas simulation `QNYX-FIRE-TALAMI-001` explores a 12-hour hypothesis under unvalidated ignition scenario assumptions.

### 2.4 Scott & Burgan Standard Fuel Models (40 Models)
- **Reference**: Scott, J.H., & Burgan, R.E. (2005). *Standard Fire Behavior Fuel Models: A Comprehensive Set for Use with Rothermel's Surface Fire Spread Model*. USDA Forest Service RMRS-GTR-153.
- **License**: Public Domain (United States Department of Agriculture).
- **Catalog File**: `configs/fuel_models_scott_burgan_qnyx.json`
- **Domain Coverage**: 100% lookup coverage across all fuel codes observed in the Talamí domain. Dynamic herbaceous fuel transfer is applied dynamically based on dead fuel moisture.

---

## 3. Data Integrity & Verification

All data files can be verified against their official release hashes using:

```bash
python scripts/download_external_data.py --verify
```

Or using the standard checksum utility:

```bash
sha256sum -c manifests/sha256_checksums.txt
```
