# sarenv/swarm/environment.py
"""
Wrapper del grid sobre SARDatasetItem para el simulador de enjambre.

Expone el mapa de probabilidad, mapas de terreno (detección y transitabilidad)
y funciones de conversión mundo <-> grid. El campo estigmérgico de presencia
lo mantiene cada agente en su ``LocalKnowledgeMap`` (estigmergia swarm-local,
propagada por gossip), no el entorno.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .terrain import (
    _rasterize_features,
    build_detection_modifier_map,
    build_traversability_map,
)

if TYPE_CHECKING:
    from sarenv.core.loading import SARDatasetItem


@dataclass(frozen=True)
class GridInfo:
    """Geometría precomputada del grid para conversiones rápidas mundo <-> grid."""

    rows: int
    cols: int
    minx: float
    miny: float
    maxx: float
    maxy: float
    dx: float  # ancho de celda en metros
    dy: float  # alto de celda en metros


class SwarmEnvironment:
    """Wrapper sobre SARDatasetItem que da acceso grid al mapa de probabilidad.

    Convierte el dataset en una representación discreta con la que los agentes
    del enjambre pueden operar directamente (coordenadas grid, vecinos, costes).
    """

    def __init__(self, dataset_item: SARDatasetItem) -> None:
        self.dataset_item = dataset_item
        # Normalizar a [0, 1] para que el scoring no esté dominado por
        # la escala absoluta del heatmap (cuyos valores son ~1e-5).
        # Guardamos el heatmap original para PathEvaluator (métricas).
        self.raw_heatmap: np.ndarray = dataset_item.heatmap.astype(np.float32)
        pmax = self.raw_heatmap.max()
        self.probability_map: np.ndarray = (
            self.raw_heatmap / pmax if pmax > 0 else self.raw_heatmap.copy()
        )
        self.bounds: tuple[float, float, float, float] = dataset_item.bounds

        rows, cols = self.probability_map.shape
        minx, miny, maxx, maxy = self.bounds
        dx = (maxx - minx) / cols
        dy = (maxy - miny) / rows

        self.grid = GridInfo(
            rows=rows, cols=cols,
            minx=minx, miny=miny, maxx=maxx, maxy=maxy,
            dx=dx, dy=dy,
        )

        # Centro del entorno en coordenadas mundo (para despliegue inicial)
        self.center_x: float = (minx + maxx) / 2.0
        self.center_y: float = (miny + maxy) / 2.0

        # -- Mapas de terreno por tipo de agente --
        # Rasterizar features vectoriales a grid de tipo de terreno
        self.terrain_grid: np.ndarray = _rasterize_features(
            dataset_item.features, rows, cols, minx, miny, dx, dy,
        )

        # Mapas de modificadores por tipo de agente (cacheados en dict)
        self._detection_modifier: dict[str, np.ndarray] = {}
        self._traversability: dict[str, np.ndarray] = {}

    def get_detection_modifier(self, agent_type: str) -> np.ndarray:
        """Grid float32 de modificadores de detección para el tipo de agente.

        Se cachea tras la primera llamada.
        """
        if agent_type not in self._detection_modifier:
            self._detection_modifier[agent_type] = build_detection_modifier_map(
                self.terrain_grid, agent_type,
            )
        return self._detection_modifier[agent_type]

    def get_traversability(self, agent_type: str) -> np.ndarray:
        """Grid float32 de costes de transitabilidad para el tipo de agente.

        Se cachea tras la primera llamada.
        """
        if agent_type not in self._traversability:
            self._traversability[agent_type] = build_traversability_map(
                self.terrain_grid, agent_type,
            )
        return self._traversability[agent_type]

    # -- Conversión de coordenadas --

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """Coordenadas mundo (x, y) -> índices grid (row, col)."""
        col = int((x - self.grid.minx) / self.grid.dx)
        row = int((y - self.grid.miny) / self.grid.dy)
        col = max(0, min(col, self.grid.cols - 1))
        row = max(0, min(row, self.grid.rows - 1))
        return row, col

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        """Indices grid (row, col) -> coordenadas mundo del centro de la celda."""
        x = self.grid.minx + (col + 0.5) * self.grid.dx
        y = self.grid.miny + (row + 0.5) * self.grid.dy
        return x, y

    def in_bounds(self, row: int, col: int) -> bool:
        """True si la celda está dentro de los límites del mapa."""
        return 0 <= row < self.grid.rows and 0 <= col < self.grid.cols

    # -- Detección: circular uniforme, como el greedy de SAREnv --

    def get_visible_cells(
        self,
        row: int,
        col: int,
        detection_radius: float,
    ) -> set[tuple[int, int]]:
        """Celdas visibles desde (row, col) con footprint circular.

        Optimización: calcula distancias en espacio grid escalado en vez de
        convertir cada celda a coordenadas mundo (evita N² llamadas a grid_to_world).
        """
        # Radio en número de celdas por cada eje
        radius_cells_x = int(np.ceil(detection_radius / self.grid.dx))
        radius_cells_y = int(np.ceil(detection_radius / self.grid.dy))
        r2 = detection_radius * detection_radius

        visible: set[tuple[int, int]] = set()

        r_min = max(0, row - radius_cells_y)
        r_max = min(self.grid.rows, row + radius_cells_y + 1)
        c_min = max(0, col - radius_cells_x)
        c_max = min(self.grid.cols, col + radius_cells_x + 1)

        for r in range(r_min, r_max):
            dy_m = (r - row) * self.grid.dy
            dy2 = dy_m * dy_m
            # Si solo la componente Y ya excede el radio, saltar fila
            if dy2 > r2:
                continue
            for c in range(c_min, c_max):
                dx_m = (c - col) * self.grid.dx
                if dy2 + dx_m * dx_m <= r2:
                    visible.add((r, c))

        return visible

    # -- Movimiento: 8-conectado, coste modulado por terreno --

    # Offsets de los 8 vecinos
    NEIGHBOR_OFFSETS = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    def get_reachable_neighbors(
        self, row: int, col: int, agent_type: str = "drone"
    ) -> list[tuple[int, int]]:
        """Vecinos alcanzables desde (row, col) en un tick.

        Excluye celdas intransitables (coste infinito) para el tipo de agente.
        """
        trav = self.get_traversability(agent_type)
        neighbors: list[tuple[int, int]] = []
        for dr, dc in self.NEIGHBOR_OFFSETS:
            nr, nc = row + dr, col + dc
            if self.in_bounds(nr, nc) and np.isfinite(trav[nr, nc]):
                neighbors.append((nr, nc))
        return neighbors

    def movement_cost(
        self, from_rc: tuple[int, int], to_rc: tuple[int, int], agent_type: str = "drone"
    ) -> float:
        """Coste en metros de moverse entre dos celdas adyacentes.

        Distancia euclídea multiplicada por el coste de transitabilidad
        de la celda destino para el tipo de agente.
        """
        dr = to_rc[0] - from_rc[0]
        dc = to_rc[1] - from_rc[1]
        base_dist = np.sqrt((dr * self.grid.dy) ** 2 + (dc * self.grid.dx) ** 2)
        trav = self.get_traversability(agent_type)
        return base_dist * trav[to_rc[0], to_rc[1]]
