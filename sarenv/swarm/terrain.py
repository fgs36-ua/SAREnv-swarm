# sarenv/swarm/terrain.py
"""
Rasterización de features vectoriales a grids de modificadores de terreno.

Convierte el GeoDataFrame de features del dataset (polígonos, líneas, puntos)
en dos grids alineados con el heatmap:
  - detection_modifier: multiplica la calidad de detección por celda y tipo de agente
  - traversability_cost: coste de movimiento por celda y tipo de agente

Los tipos de feature vienen de OSM (road, woodland, water, etc.) y cada agente
tiene modificadores distintos según la tabla DETECTION_MODIFIERS / TRAVERSABILITY_COSTS.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from shapely.geometry import LineString, Polygon

if TYPE_CHECKING:
    import geopandas as gpd


# -- Modificadores de detección por tipo de agente y terreno --
# Valores en [0, 1]: 1.0 = detección perfecta, 0.0 = invisible

DETECTION_MODIFIERS: dict[str, dict[str, float]] = {
    "drone": {
        "field": 1.0,
        "road": 0.95,
        "brush": 0.6,
        "scrub": 0.5,
        "woodland": 0.15,    # dosel arbóreo bloquea casi toda la visión aérea
        "water": 1.0,
        "structure": 0.8,
        "rock": 0.9,
        "drainage": 0.9,
        "linear": 0.95,
    },
    "robot_dog": {
        "field": 0.7,
        "road": 0.8,
        "brush": 0.6,
        "scrub": 0.5,
        "woodland": 0.85,    # sensores térmicos/olfato a nivel suelo funcionan bien
        "water": 0.0,        # no puede estar en agua
        "structure": 0.9,
        "rock": 0.4,
        "drainage": 0.3,
        "linear": 0.8,
    },
}

# Valor por defecto para celdas sin feature reconocida (terreno abierto genérico)
_DEFAULT_DETECTION = {"drone": 1.0, "robot_dog": 0.7}

# -- Costes de transitabilidad por tipo de agente y terreno --
# Multiplicador sobre la distancia euclídea.  inf = intransitable.

TRAVERSABILITY_COSTS: dict[str, dict[str, float]] = {
    "drone": {
        # Los drones vuelan: todo el terreno cuesta igual
        "default": 1.0,
    },
    "robot_dog": {
        "road": 0.5,
        "field": 1.0,
        "brush": 2.0,
        "scrub": 1.8,
        "woodland": 1.5,
        "water": float("inf"),
        "drainage": float("inf"),
        "structure": 1.2,
        "rock": 3.0,
        "linear": 0.8,
    },
}

_DEFAULT_TRAVERSABILITY = {"drone": 1.0, "robot_dog": 1.0}


def _rasterize_features(
    features: gpd.GeoDataFrame,
    rows: int,
    cols: int,
    minx: float,
    miny: float,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Rasteriza features a un grid de strings con el tipo de terreno dominante.

    Devuelve un array de objetos (str | None) de shape (rows, cols).
    None = sin feature → terreno abierto por defecto.

    Para cada celda, si hay varias features superpuestas, gana la que
    tiene mayor área intersectada. Esto es una simplificación, pero
    funciona bien para el tamaño de celda típico (30m).
    """
    from skimage.draw import polygon as ski_polygon

    # Grid de tipo de terreno (None = sin datos = campo abierto)
    terrain_grid: np.ndarray = np.empty((rows, cols), dtype=object)

    if features is None or features.empty:
        return terrain_grid

    # Agrupar por tipo de feature para rasterizar de golpe
    if "feature_type" not in features.columns:
        return terrain_grid

    for ftype, group in features.groupby("feature_type"):
        for geom in group.geometry:
            if geom is None or geom.is_empty:
                continue

            if isinstance(geom, Polygon):
                # Rasterizar polígono entero
                coords = np.array(geom.exterior.coords)
                img_c = ((coords[:, 0] - minx) / dx).astype(int)
                img_r = ((coords[:, 1] - miny) / dy).astype(int)
                rr, cc = ski_polygon(img_r, img_c, shape=(rows, cols))
                terrain_grid[rr, cc] = ftype

            elif isinstance(geom, LineString):
                # Rasterizar línea con puntos intermedios
                coords = np.array(geom.coords)
                for k in range(len(coords) - 1):
                    x0, y0 = coords[k]
                    x1, y1 = coords[k + 1]
                    seg_len = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
                    n_pts = max(2, int(seg_len / min(dx, dy)))
                    xs = np.linspace(x0, x1, n_pts)
                    ys = np.linspace(y0, y1, n_pts)
                    cs = np.clip(((xs - minx) / dx).astype(int), 0, cols - 1)
                    rs = np.clip(((ys - miny) / dy).astype(int), 0, rows - 1)
                    terrain_grid[rs, cs] = ftype

            # Puntos: no los rasterizamos (demasiado pequeños para una celda)

    return terrain_grid


def build_detection_modifier_map(
    terrain_grid: np.ndarray,
    agent_type: str,
) -> np.ndarray:
    """Genera el grid de modificadores de detección para un tipo de agente.

    Devuelve array float32 de shape (rows, cols) con valores en [0, 1].
    """
    rows, cols = terrain_grid.shape
    modifiers = DETECTION_MODIFIERS.get(agent_type, {})
    default = _DEFAULT_DETECTION.get(agent_type, 1.0)

    result = np.full((rows, cols), default, dtype=np.float32)

    for ftype, mod_val in modifiers.items():
        mask = terrain_grid == ftype
        result[mask] = mod_val

    return result


def build_traversability_map(
    terrain_grid: np.ndarray,
    agent_type: str,
) -> np.ndarray:
    """Genera el grid de costes de transitabilidad para un tipo de agente.

    Devuelve array float32 de shape (rows, cols) con multiplicadores >= 0.
    inf = intransitable.
    """
    rows, cols = terrain_grid.shape
    costs = TRAVERSABILITY_COSTS.get(agent_type, {})
    default = costs.get("default", _DEFAULT_TRAVERSABILITY.get(agent_type, 1.0))

    result = np.full((rows, cols), default, dtype=np.float32)

    for ftype, cost_val in costs.items():
        if ftype == "default":
            continue
        mask = terrain_grid == ftype
        result[mask] = cost_val

    return result
