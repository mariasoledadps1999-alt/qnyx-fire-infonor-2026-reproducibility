"""
src/fire_engine/fuels/models.py

Estructuras de datos tipadas para modelos de combustible de Rothermel (1972 INT-115 / Albini 1976 INT-30).
Incluye los modelos iniciales canónicos 1, 4 y 8.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from ..units import TONS_ACRE_TO_LB_FT2, FEET_TO_METERS, METERS_TO_FEET, LB_FT2_TO_KG_M2


@dataclass(frozen=True)
class FuelParticleClass:
    """
    Representa un estrato/clase de partícula de combustible en el lecho de Rothermel.
    """
    name: str
    load_lb_ft2: float              # Carga de combustible seco w0 [lb/ft²]
    sav_inv_ft: float               # Relación Superficie/Volumen sigma [1/ft]
    heat_btu_lb: float = 8000.0     # Poder calorífico bajo h [Btu/lb]
    is_live: bool = False           # ¿Es combustible vivo?
    is_herbaceous: bool = False     # ¿Es herbáceo vivo?

    @property
    def load_kg_m2(self) -> float:
        return self.load_lb_ft2 * LB_FT2_TO_KG_M2

    @property
    def sav_inv_m(self) -> float:
        return self.sav_inv_ft / FEET_TO_METERS


@dataclass(frozen=True)
class RothermelFuelBed:
    """
    Representa un lecho de combustible completo estratificado para el kernel de Rothermel.
    """
    model_id: int
    name: str
    description: str
    depth_ft: float                                     # Profundidad del lecho delta [ft]
    dead_extinction_moisture: float                     # Humedad de extinción del combustible muerto Mx (fracción)
    set_name: str = "ORIGINAL_13"                       # Conjunto de catálogo ("ORIGINAL_13", "SCOTT_BURGAN_40")
    total_mineral_content: float = 0.0555               # Contenido mineral total ST (fracción)
    effective_mineral_content: float = 0.010            # Contenido de sílice libre Se (fracción)
    particle_density_lb_ft3: float = 32.0               # Densidad de partícula rho_p [lb/ft³]
    particle_classes: List[FuelParticleClass] = field(default_factory=list)
    is_burnable: bool = True
    is_dynamic: bool = False

    @property
    def depth_m(self) -> float:
        return self.depth_ft * FEET_TO_METERS

    @property
    def total_load_lb_ft2(self) -> float:
        return sum(p.load_lb_ft2 for p in self.particle_classes)

    @property
    def total_load_kg_m2(self) -> float:
        return self.total_load_lb_ft2 * LB_FT2_TO_KG_M2

    @property
    def dead_load_lb_ft2(self) -> float:
        return sum(p.load_lb_ft2 for p in self.particle_classes if not p.is_live)

    @property
    def live_load_lb_ft2(self) -> float:
        return sum(p.load_lb_ft2 for p in self.particle_classes if p.is_live)


@dataclass(frozen=True)
class FuelMoisture:
    """
    Contenido de humedad de cada fracción de combustible (en fracción 0.0 - 1.0 o > 1.0 para vivos).
    """
    dead_1h: float                  # Humedad muerto 1-h (0 a 0.25")
    dead_10h: float = 0.0           # Humedad muerto 10-h (0.25 a 1.0")
    dead_100h: float = 0.0          # Humedad muerto 100-h (1.0 a 3.0")
    live_herbaceous: float = 0.0    # Humedad vivo herbáceo
    live_woody: float = 0.0         # Humedad vivo leñoso

    def get_moisture_for_class(self, p_class: FuelParticleClass) -> float:
        """Devuelve la humedad correspondiente a una clase de partícula específica."""
        if not p_class.is_live:
            if (
                "dead_herb" in p_class.name.lower()
                or "herb" in p_class.name.lower()
                or p_class.sav_inv_ft >= 1200.0
                or "1h" in p_class.name.lower()
                or "1-h" in p_class.name.lower()
            ):
                return self.dead_1h
            elif p_class.sav_inv_ft >= 100.0 or "10h" in p_class.name.lower() or "10-h" in p_class.name.lower():
                return self.dead_10h
            else:
                return self.dead_100h
        else:
            if p_class.is_herbaceous:
                return self.live_herbaceous
            else:
                return self.live_woody


# =============================================================================
# CATÁLOGO DE LOS 13 MODELOS ESTÁNDAR ORIGINALES (ALBINI 1976 / ANDERSON 1982)
# =============================================================================

ORIGINAL_13_FUEL_MODELS: Dict[int, RothermelFuelBed] = {
    # -------------------------------------------------------------------------
    # Modelo 1: Short Grass (Pastizal Corto)
    # Fuente: Albini 1976 INT-30 p. 69 / Anderson 1982 INT-122 p. 4
    # -------------------------------------------------------------------------
    1: RothermelFuelBed(
        model_id=1,
        name="Short Grass",
        description="Short grass (1 ft depth), fine fuels, very high rate of spread",
        depth_ft=1.0,
        dead_extinction_moisture=0.12,
        set_name="ORIGINAL_13",
        particle_classes=[
            FuelParticleClass("dead_1h", 0.74 * TONS_ACRE_TO_LB_FT2, 3500.0, 8000.0, False, False)
        ]
    ),

    # -------------------------------------------------------------------------
    # Modelo 2: Timber (Grass and Understory)
    # Fuente: Albini 1976 INT-30 p. 70 / Anderson 1982 INT-122 p. 5
    # -------------------------------------------------------------------------
    2: RothermelFuelBed(
        model_id=2,
        name="Timber (Grass and Understory)",
        description="Timber with grass and herbaceous understory, surface fire",
        depth_ft=1.0,
        dead_extinction_moisture=0.15,
        set_name="ORIGINAL_13",
        particle_classes=[
            FuelParticleClass("dead_1h", 2.00 * TONS_ACRE_TO_LB_FT2, 3000.0, 8000.0, False, False),
            FuelParticleClass("dead_10h", 1.00 * TONS_ACRE_TO_LB_FT2, 109.0, 8000.0, False, False),
            FuelParticleClass("dead_100h", 0.50 * TONS_ACRE_TO_LB_FT2, 30.0, 8000.0, False, False),
            FuelParticleClass("live_herbaceous", 0.50 * TONS_ACRE_TO_LB_FT2, 1500.0, 8000.0, True, True)
        ]
    ),

    # -------------------------------------------------------------------------
    # Modelo 3: Tall Grass (Pastizal Alto)
    # Fuente: Albini 1976 INT-30 p. 71 / Anderson 1982 INT-122 p. 6
    # -------------------------------------------------------------------------
    3: RothermelFuelBed(
        model_id=3,
        name="Tall Grass",
        description="Tall grass (2.5 ft depth), heavy loadings, high rate of spread",
        depth_ft=2.5,
        dead_extinction_moisture=0.25,
        set_name="ORIGINAL_13",
        particle_classes=[
            FuelParticleClass("dead_1h", 3.01 * TONS_ACRE_TO_LB_FT2, 1500.0, 8000.0, False, False)
        ]
    ),

    # -------------------------------------------------------------------------
    # Modelo 4: Chaparral / High Load Shrub (Matorral Denso Alto)
    # Fuente: Albini 1976 INT-30 p. 72 / Anderson 1982 INT-122 p. 7
    # -------------------------------------------------------------------------
    4: RothermelFuelBed(
        model_id=4,
        name="Chaparral / High Load Shrub",
        description="Tall chaparral (6 ft depth), heavy loadings, extremely intense fire",
        depth_ft=6.0,
        dead_extinction_moisture=0.20,
        set_name="ORIGINAL_13",
        particle_classes=[
            FuelParticleClass("dead_1h", 5.01 * TONS_ACRE_TO_LB_FT2, 2000.0, 8000.0, False, False),
            FuelParticleClass("dead_10h", 4.01 * TONS_ACRE_TO_LB_FT2, 109.0, 8000.0, False, False),
            FuelParticleClass("dead_100h", 2.01 * TONS_ACRE_TO_LB_FT2, 30.0, 8000.0, False, False),
            FuelParticleClass("live_woody", 5.01 * TONS_ACRE_TO_LB_FT2, 1500.0, 8000.0, True, False)
        ]
    ),

    # -------------------------------------------------------------------------
    # Modelo 5: Brush / Light Load Shrub (Matorral Bajo Ligero)
    # Fuente: Albini 1976 INT-30 p. 73 / Anderson 1982 INT-122 p. 8
    # -------------------------------------------------------------------------
    5: RothermelFuelBed(
        model_id=5,
        name="Brush / Light Load Shrub",
        description="Low brush (2 ft depth), light fuel load, active surface fire",
        depth_ft=2.0,
        dead_extinction_moisture=0.20,
        set_name="ORIGINAL_13",
        particle_classes=[
            FuelParticleClass("dead_1h", 1.00 * TONS_ACRE_TO_LB_FT2, 2000.0, 8000.0, False, False),
            FuelParticleClass("dead_10h", 0.50 * TONS_ACRE_TO_LB_FT2, 109.0, 8000.0, False, False),
            FuelParticleClass("live_woody", 2.00 * TONS_ACRE_TO_LB_FT2, 1500.0, 8000.0, True, False)
        ]
    ),

    # -------------------------------------------------------------------------
    # Modelo 6: Dormant Brush / Hardwood Slash (Matorral Latente)
    # Fuente: Albini 1976 INT-30 p. 74 / Anderson 1982 INT-122 p. 9
    # -------------------------------------------------------------------------
    6: RothermelFuelBed(
        model_id=6,
        name="Dormant Brush / Hardwood Slash",
        description="Dormant brush (2.5 ft depth), moderate intensity fire",
        depth_ft=2.5,
        dead_extinction_moisture=0.25,
        set_name="ORIGINAL_13",
        particle_classes=[
            FuelParticleClass("dead_1h", 1.50 * TONS_ACRE_TO_LB_FT2, 1750.0, 8000.0, False, False),
            FuelParticleClass("dead_10h", 2.50 * TONS_ACRE_TO_LB_FT2, 109.0, 8000.0, False, False),
            FuelParticleClass("dead_100h", 2.00 * TONS_ACRE_TO_LB_FT2, 30.0, 8000.0, False, False)
        ]
    ),

    # -------------------------------------------------------------------------
    # Modelo 7: Southern Rough (Matorral del Sur / Palmetto)
    # Fuente: Albini 1976 INT-30 p. 75 / Anderson 1982 INT-122 p. 10
    # -------------------------------------------------------------------------
    7: RothermelFuelBed(
        model_id=7,
        name="Southern Rough",
        description="Southern rough (2.5 ft depth), high moisture extinction (40%)",
        depth_ft=2.5,
        dead_extinction_moisture=0.40,
        set_name="ORIGINAL_13",
        particle_classes=[
            FuelParticleClass("dead_1h", 1.13 * TONS_ACRE_TO_LB_FT2, 1750.0, 8000.0, False, False),
            FuelParticleClass("dead_10h", 1.87 * TONS_ACRE_TO_LB_FT2, 109.0, 8000.0, False, False),
            FuelParticleClass("dead_100h", 1.50 * TONS_ACRE_TO_LB_FT2, 30.0, 8000.0, False, False),
            FuelParticleClass("live_woody", 0.37 * TONS_ACRE_TO_LB_FT2, 1550.0, 8000.0, True, False)
        ]
    ),

    # -------------------------------------------------------------------------
    # Modelo 8: Closed Timber Litter (Hojarasca Compacta de Bosque Cerrado)
    # Fuente: Albini 1976 INT-30 p. 76 / Anderson 1982 INT-122 p. 11
    # -------------------------------------------------------------------------
    8: RothermelFuelBed(
        model_id=8,
        name="Closed Timber Litter",
        description="Slow burning ground fire with low flame lengths under closed canopy",
        depth_ft=0.2,
        dead_extinction_moisture=0.30,
        set_name="ORIGINAL_13",
        particle_classes=[
            FuelParticleClass("dead_1h", 1.50 * TONS_ACRE_TO_LB_FT2, 2000.0, 8000.0, False, False),
            FuelParticleClass("dead_10h", 1.00 * TONS_ACRE_TO_LB_FT2, 109.0, 8000.0, False, False),
            FuelParticleClass("dead_100h", 2.50 * TONS_ACRE_TO_LB_FT2, 30.0, 8000.0, False, False)
        ]
    ),

    # -------------------------------------------------------------------------
    # Modelo 9: Hardwood Litter / Longneedle Pine (Hojarasca Suelta de Frondosas)
    # Fuente: Albini 1976 INT-30 p. 77 / Anderson 1982 INT-122 p. 12
    # -------------------------------------------------------------------------
    9: RothermelFuelBed(
        model_id=9,
        name="Hardwood Litter",
        description="Loose hardwood or longneedle pine litter, fast surface spread",
        depth_ft=0.2,
        dead_extinction_moisture=0.25,
        set_name="ORIGINAL_13",
        particle_classes=[
            FuelParticleClass("dead_1h", 2.92 * TONS_ACRE_TO_LB_FT2, 2500.0, 8000.0, False, False),
            FuelParticleClass("dead_10h", 0.41 * TONS_ACRE_TO_LB_FT2, 109.0, 8000.0, False, False),
            FuelParticleClass("dead_100h", 0.15 * TONS_ACRE_TO_LB_FT2, 30.0, 8000.0, False, False)
        ]
    ),

    # -------------------------------------------------------------------------
    # Modelo 10: Timber with Litter and Understory (Bosque con Matorral y Restos)
    # Fuente: Albini 1976 INT-30 p. 78 / Anderson 1982 INT-122 p. 13
    # -------------------------------------------------------------------------
    10: RothermelFuelBed(
        model_id=10,
        name="Timber (Litter and Understory)",
        description="Forest with heavy litter and understory, active surface fire",
        depth_ft=1.0,
        dead_extinction_moisture=0.25,
        set_name="ORIGINAL_13",
        particle_classes=[
            FuelParticleClass("dead_1h", 3.01 * TONS_ACRE_TO_LB_FT2, 2000.0, 8000.0, False, False),
            FuelParticleClass("dead_10h", 2.00 * TONS_ACRE_TO_LB_FT2, 109.0, 8000.0, False, False),
            FuelParticleClass("dead_100h", 5.01 * TONS_ACRE_TO_LB_FT2, 30.0, 8000.0, False, False),
            FuelParticleClass("live_woody", 2.00 * TONS_ACRE_TO_LB_FT2, 1500.0, 8000.0, True, False)
        ]
    ),

    # -------------------------------------------------------------------------
    # Modelo 11: Light Logging Slash (Restos Ligeros de Corta)
    # Fuente: Albini 1976 INT-30 p. 79 / Anderson 1982 INT-122 p. 14
    # -------------------------------------------------------------------------
    11: RothermelFuelBed(
        model_id=11,
        name="Light Logging Slash",
        description="Light logging slash (1 ft depth), active fire spread",
        depth_ft=1.0,
        dead_extinction_moisture=0.15,
        set_name="ORIGINAL_13",
        particle_classes=[
            FuelParticleClass("dead_1h", 1.50 * TONS_ACRE_TO_LB_FT2, 1500.0, 8000.0, False, False),
            FuelParticleClass("dead_10h", 4.51 * TONS_ACRE_TO_LB_FT2, 109.0, 8000.0, False, False),
            FuelParticleClass("dead_100h", 5.51 * TONS_ACRE_TO_LB_FT2, 30.0, 8000.0, False, False)
        ]
    ),

    # -------------------------------------------------------------------------
    # Modelo 12: Medium Logging Slash (Restos Medios de Corta)
    # Fuente: Albini 1976 INT-30 p. 79 / Anderson 1982 INT-122 p. 15
    # -------------------------------------------------------------------------
    12: RothermelFuelBed(
        model_id=12,
        name="Medium Logging Slash",
        description="Medium logging slash (2.3 ft depth), rapid fire spread and high heat",
        depth_ft=2.3,
        dead_extinction_moisture=0.20,
        set_name="ORIGINAL_13",
        particle_classes=[
            FuelParticleClass("dead_1h", 4.01 * TONS_ACRE_TO_LB_FT2, 1500.0, 8000.0, False, False),
            FuelParticleClass("dead_10h", 14.03 * TONS_ACRE_TO_LB_FT2, 109.0, 8000.0, False, False),
            FuelParticleClass("dead_100h", 16.53 * TONS_ACRE_TO_LB_FT2, 30.0, 8000.0, False, False)
        ]
    ),

    # -------------------------------------------------------------------------
    # Modelo 13: Heavy Logging Slash (Restos Pesados de Corta)
    # Fuente: Albini 1976 INT-30 p. 80 / Anderson 1982 INT-122 p. 16
    # -------------------------------------------------------------------------
    13: RothermelFuelBed(
        model_id=13,
        name="Heavy Logging Slash",
        description="Heavy logging slash (3 ft depth), very intense and continuous fire",
        depth_ft=3.0,
        dead_extinction_moisture=0.25,
        set_name="ORIGINAL_13",
        particle_classes=[
            FuelParticleClass("dead_1h", 7.01 * TONS_ACRE_TO_LB_FT2, 1500.0, 8000.0, False, False),
            FuelParticleClass("dead_10h", 23.04 * TONS_ACRE_TO_LB_FT2, 109.0, 8000.0, False, False),
            FuelParticleClass("dead_100h", 28.05 * TONS_ACRE_TO_LB_FT2, 30.0, 8000.0, False, False)
        ]
    )
}

# Alias canónico
STANDARD_FUEL_MODELS: Dict[int, RothermelFuelBed] = ORIGINAL_13_FUEL_MODELS


# =============================================================================
# TRANSFERENCIA DINÁMICA DE COMBUSTIBLE HERBÁCEO (SCOTT & BURGAN 2005)
# =============================================================================

def apply_dynamic_fuel_transfer(fuel_bed: RothermelFuelBed, moisture: FuelMoisture) -> RothermelFuelBed:
    """
    Aplica la transferencia dinámica de combustible herbáceo según Scott & Burgan (2005 RMRS-GTR-153 pp. 7-8).

    Cuando la humedad del combustible herbáceo vivo (M_lh) está entre 30% (0.30) y 120% (1.20),
    una fracción se transfiere linealmente de la categoría viva a la categoría muerta fina (1-h):
      fracción transferida = (1.20 - M_lh) / (1.20 - 0.30) = (1.20 - M_lh) / 0.90

    - Si M_lh >= 1.20: fracción = 0.0 (totalmente verde)
    - Si M_lh <= 0.30: fracción = 1.0 (totalmente curado)

    El combustible transferido se evalúa a la humedad de combustible muerto 1-h, manteniendo su SAV y calor.
    """
    if not fuel_bed.is_dynamic:
        return fuel_bed

    herb_classes = [p for p in fuel_bed.particle_classes if p.is_live and p.is_herbaceous]
    if not herb_classes:
        return fuel_bed

    m_lh = moisture.live_herbaceous
    if m_lh >= 1.20:
        transfer_fraction = 0.0
    elif m_lh <= 0.30:
        transfer_fraction = 1.0
    else:
        transfer_fraction = (1.20 - m_lh) / 0.90

    if transfer_fraction <= 0.0:
        return fuel_bed

    new_particle_classes: List[FuelParticleClass] = []
    for p in fuel_bed.particle_classes:
        if p.is_live and p.is_herbaceous:
            transferred_load = p.load_lb_ft2 * transfer_fraction
            remaining_load = p.load_lb_ft2 * (1.0 - transfer_fraction)

            if transferred_load > 0.0:
                new_particle_classes.append(
                    FuelParticleClass(
                        name="dead_herbaceous",
                        load_lb_ft2=transferred_load,
                        sav_inv_ft=p.sav_inv_ft,
                        heat_btu_lb=p.heat_btu_lb,
                        is_live=False,
                        is_herbaceous=False
                    )
                )
            if remaining_load > 0.0:
                new_particle_classes.append(
                    FuelParticleClass(
                        name=p.name,
                        load_lb_ft2=remaining_load,
                        sav_inv_ft=p.sav_inv_ft,
                        heat_btu_lb=p.heat_btu_lb,
                        is_live=True,
                        is_herbaceous=True
                    )
                )
        else:
            new_particle_classes.append(p)

    return RothermelFuelBed(
        model_id=fuel_bed.model_id,
        name=fuel_bed.name,
        description=fuel_bed.description,
        depth_ft=fuel_bed.depth_ft,
        dead_extinction_moisture=fuel_bed.dead_extinction_moisture,
        set_name=fuel_bed.set_name,
        total_mineral_content=fuel_bed.total_mineral_content,
        effective_mineral_content=fuel_bed.effective_mineral_content,
        particle_density_lb_ft3=fuel_bed.particle_density_lb_ft3,
        particle_classes=new_particle_classes,
        is_burnable=fuel_bed.is_burnable,
        is_dynamic=fuel_bed.is_dynamic
    )


# =============================================================================
# CATÁLOGO SCOTT & BURGAN 40 ESTÁNDAR (RMRS-GTR-153)
# =============================================================================

def load_scott_burgan_catalog(catalog_path: Optional[str] = None) -> Dict[int, RothermelFuelBed]:
    """
    Carga el catálogo estándar Scott & Burgan 40 desde el archivo JSON de configuración.
    """
    import json
    import os

    if catalog_path is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        catalog_path = os.path.join(base_dir, "configs", "fuel_models_scott_burgan_qnyx.json")

    if not os.path.exists(catalog_path):
        cwd_path = os.path.join("configs", "fuel_models_scott_burgan_qnyx.json")
        if os.path.exists(cwd_path):
            catalog_path = cwd_path
        else:
            raise FileNotFoundError(f"No se encontró el catálogo Scott & Burgan en {catalog_path}")

    with open(catalog_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    catalog: Dict[int, RothermelFuelBed] = {}
    for code_str, m in data["models"].items():
        code = int(code_str)
        p_classes = []
        for p in m.get("particle_classes", []):
            load_lb_ft2 = p["load_tons_acre"] * TONS_ACRE_TO_LB_FT2
            p_classes.append(
                FuelParticleClass(
                    name=p["name"],
                    load_lb_ft2=load_lb_ft2,
                    sav_inv_ft=float(p["sav_inv_ft"]),
                    heat_btu_lb=float(p.get("heat_btu_lb", 8000.0)),
                    is_live=bool(p.get("is_live", False)),
                    is_herbaceous=bool(p.get("is_herbaceous", False))
                )
            )

        catalog[code] = RothermelFuelBed(
            model_id=code,
            name=m["name"],
            description=m.get("description", ""),
            depth_ft=float(m.get("depth_ft", 0.0)),
            dead_extinction_moisture=float(m.get("dead_extinction_moisture", 0.0)),
            set_name="SCOTT_BURGAN_STANDARD_FUEL",
            total_mineral_content=float(m.get("total_mineral_content", 0.0555)),
            effective_mineral_content=float(m.get("effective_mineral_content", 0.010)),
            particle_density_lb_ft3=float(m.get("particle_density_lb_ft3", 32.0)),
            particle_classes=p_classes,
            is_burnable=bool(m.get("burnable", True)),
            is_dynamic=bool(m.get("dynamic", False))
        )

    return catalog


# Catálogo Scott & Burgan perezoso o precargado
try:
    SCOTT_BURGAN_STANDARD_FUEL_MODELS: Dict[int, RothermelFuelBed] = load_scott_burgan_catalog()
except Exception:
    SCOTT_BURGAN_STANDARD_FUEL_MODELS = {}


def get_standard_fuel_model(model_id: int, set_name: str = "ORIGINAL_13") -> RothermelFuelBed:
    """
    Obtiene una copia inmutable del lecho de combustible estándar por ID y conjunto de catálogo.
    Soporta 'ORIGINAL_13' y 'SCOTT_BURGAN_STANDARD_FUEL' / 'SCOTT_BURGAN_40'.
    """
    if set_name == "ORIGINAL_13":
        catalog = ORIGINAL_13_FUEL_MODELS
    elif set_name in ("SCOTT_BURGAN_STANDARD_FUEL", "SCOTT_BURGAN_40", "SCOTT_BURGAN"):
        global SCOTT_BURGAN_STANDARD_FUEL_MODELS
        if not SCOTT_BURGAN_STANDARD_FUEL_MODELS:
            SCOTT_BURGAN_STANDARD_FUEL_MODELS = load_scott_burgan_catalog()
        catalog = SCOTT_BURGAN_STANDARD_FUEL_MODELS
    else:
        raise ValueError(
            f"Conjunto de catálogo '{set_name}' no soportado. "
            f"Disponibles: ['ORIGINAL_13', 'SCOTT_BURGAN_STANDARD_FUEL', 'SCOTT_BURGAN_40']"
        )

    if model_id not in catalog:
        raise ValueError(
            f"Modelo de combustible ID={model_id} no encontrado en el catálogo {set_name}. "
            f"Disponibles: {sorted(list(catalog.keys()))}"
        )
    return catalog[model_id]
