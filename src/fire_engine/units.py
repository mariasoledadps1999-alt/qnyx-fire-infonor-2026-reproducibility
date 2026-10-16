"""
src/fire_engine/units.py

Constantes nombradas y factores de conversión de unidades entre Sistema Internacional (SI)
y Unidades Nativo-Inglesas utilizadas por el modelo de propagación de Rothermel (1972 INT-115).
"""

# =============================================================================
# Longitud
# =============================================================================
FEET_TO_METERS: float = 0.3048
METERS_TO_FEET: float = 1.0 / 0.3048  # ≈ 3.280839895013123

INCHES_TO_CM: float = 2.54
CM_TO_INCHES: float = 1.0 / 2.54

# =============================================================================
# Velocidad y Tasa de Propagación (Rate of Spread)
# =============================================================================
FT_MIN_TO_M_MIN: float = 0.3048
M_MIN_TO_FT_MIN: float = 1.0 / 0.3048  # ≈ 3.280839895013123

FT_MIN_TO_M_S: float = 0.3048 / 60.0   # = 0.00508
M_S_TO_FT_MIN: float = 60.0 / 0.3048   # ≈ 196.8503937007874

M_S_TO_MPH: float = 2.2369362920544023
MPH_TO_M_S: float = 0.44704

MPH_TO_FT_MIN: float = 88.0
FT_MIN_TO_MPH: float = 1.0 / 88.0      # ≈ 0.011363636363636364

CHAINS_HR_TO_FT_MIN: float = 66.0 / 60.0  # = 1.1
FT_MIN_TO_CHAINS_HR: float = 60.0 / 66.0  # ≈ 0.9090909090909091

CHAINS_HR_TO_M_MIN: float = 66.0 * 0.3048 / 60.0  # = 0.33528
M_MIN_TO_CHAINS_HR: float = 60.0 / (66.0 * 0.3048)  # ≈ 2.982581722739203

# =============================================================================
# Masa y Carga de Combustible (Fuel Loading)
# =============================================================================
TONS_ACRE_TO_LB_FT2: float = 2000.0 / 43560.0  # ≈ 0.04591368227731864
LB_FT2_TO_TONS_ACRE: float = 43560.0 / 2000.0  # = 21.78

LB_FT2_TO_KG_M2: float = 4.88242763
KG_M2_TO_LB_FT2: float = 1.0 / 4.88242763      # ≈ 0.204816143

TONS_ACRE_TO_KG_M2: float = (2000.0 / 43560.0) * 4.88242763  # ≈ 0.224170231
KG_M2_TO_TONS_ACRE: float = 1.0 / TONS_ACRE_TO_KG_M2         # ≈ 4.46089559

# =============================================================================
# Densidad Volumétrica
# =============================================================================
LB_FT3_TO_KG_M3: float = 16.01846337
KG_M3_TO_LB_FT3: float = 1.0 / 16.01846337     # ≈ 0.06242796

# =============================================================================
# Relación Superficie / Volumen (Surface Area-to-Volume Ratio - SAV, sigma)
# =============================================================================
INV_FT_TO_INV_M: float = 1.0 / 0.3048          # ≈ 3.280839895
INV_M_TO_INV_FT: float = 0.3048

INV_FT_TO_INV_CM: float = 1.0 / 30.48          # ≈ 0.03280839895
INV_CM_TO_INV_FT: float = 30.48

# =============================================================================
# Energía e Intensidad
# =============================================================================
BTU_LB_TO_KJ_KG: float = 2.326
KJ_KG_TO_BTU_LB: float = 1.0 / 2.326           # ≈ 0.42992261

BTU_FT2_MIN_TO_KW_M2: float = 0.1891435
KW_M2_TO_BTU_FT2_MIN: float = 1.0 / 0.1891435  # ≈ 5.286991
