# sarenv/swarm/knowledge.py
"""
Mapa de conocimiento local: cada agente mantiene su propia visión del mundo.

En Fase 1 todos comparten el mismo mapa (max_hops=∞) para poder validar
contra el algoritmo greedy existente. En Fase 3 cada agente tendrá su
copia local y usará gossip para propagar actualizaciones.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np


@dataclass
class MapUpdate:
    """Actualización atómica de una celda en una capa de feromonas."""

    cell: tuple[int, int]  # (row, col)
    layer: str  # "exploration" | "alert"
    value: float  # intensidad de feromona
    timestamp: int  # tick en que se generó
    origin_agent: str  # id del agente que la originó
    hops: int = 0  # nº de saltos (para limitar propagación)


# Límite de entradas en _latest_updates para evitar consumo excesivo de memoria
# en simulaciones largas.  Cuando se supera, se podan las más antiguas.
_UPDATE_BUFFER_LIMIT = 50_000


class LocalKnowledgeMap:
    """Mapa de conocimiento por agente.

    Contiene tres capas:
    - probability_map  -- copia estática del heatmap inicial
    - exploration_map  -- feromona de exploración ("ya pasé por aquí")
    - alert_map        -- feromona de alerta ("aquí hay algo interesante")
    """

    def __init__(self, probability_map: np.ndarray) -> None:
        # Probabilidad de referencia (se copia, cada agente tiene la suya)
        # Normalizamos a [0, 1] para que prob * novelty compita con repulsión
        # en la función de score.  Los valores raw son ~1e-5 (distribución sobre
        # todo el grid) y la repulsión es ~0.01, así que sin normalizar los
        # agentes caen siempre a random walk.
        raw = probability_map.copy().astype(np.float32)
        pmax = raw.max()
        if pmax > 0:
            raw /= pmax
        self.probability_map = raw

        # Feromona de exploración: 0.0 = sin explorar, 1.0 = explorado al 100%
        self.exploration_map = np.zeros_like(self.probability_map, dtype=np.float32)

        # Feromona de alerta: 0.0 = nada, cuanto mayor, más interesante
        self.alert_map = np.zeros_like(self.probability_map, dtype=np.float32)

        # Feromona de presencia LOCAL (estigmergia swarm pura, sin estado
        # global en el entorno): cada agente mantiene su propia copia,
        # deposita en su celda actual cada tick y la evapora/difunde por
        # su cuenta. El gossip propaga el campo a otros agentes en rango
        # mediante merge ``np.maximum`` (info-pesimista: si alguno sabe
        # que una zona está pisada, los demás también).
        self.presence_field = np.zeros_like(self.probability_map, dtype=np.float32)

        # Diccionario de celdas exploradas conocidas (propias + gossip).
        # Mapea celda -> timestamp de última observación conocida.
        # Caduca tras GOSSIP_EXPIRY_TICKS para permitir re-exploración a largo plazo.
        self.cells_gossip_explored: Dict[Tuple[int, int], int] = {}
        self.gossip_expiry_ticks: int = 15_000

        # Buffer de actualizaciones indexado por (row, col, layer)
        self._latest_updates: Dict[Tuple[int, int, str], MapUpdate] = {}

    # -- Observación --

    def record_observation(
        self,
        visible_cells: set[tuple[int, int]],
        agent_id: str,
        timestep: int,
        detection_quality: float = 1.0,
    ) -> None:
        """Marca las celdas visibles como exploradas en exploration_map.

        Solo actualiza si la nueva calidad de detección supera la registrada.
        """
        for r, c in visible_cells:
            self.cells_gossip_explored[(r, c)] = timestep
            effective = detection_quality
            if effective > self.exploration_map[r, c]:
                self.exploration_map[r, c] = effective
                update = MapUpdate(
                    cell=(r, c),
                    layer="exploration",
                    value=effective,
                    timestamp=timestep,
                    origin_agent=agent_id,
                    hops=0,
                )
                self._latest_updates[(r, c, "exploration")] = update

    def record_alert(
        self,
        cell: tuple[int, int],
        intensity: float,
        agent_id: str,
        timestep: int,
    ) -> None:
        """Deposita feromona de alerta en la celda indicada."""
        r, c = cell
        if intensity > self.alert_map[r, c]:
            self.alert_map[r, c] = intensity
            update = MapUpdate(
                cell=cell,
                layer="alert",
                value=intensity,
                timestamp=timestep,
                origin_agent=agent_id,
                hops=0,
            )
            self._latest_updates[(r, c, "alert")] = update

    # -- Helpers de gossip (Fase 3 los usará mucho más) --

    # ------------------------------------------------------------------
    # Feromona de presencia LOCAL (swarm puro, gossip-merged)
    # ------------------------------------------------------------------

    def deposit_presence(self, row: int, col: int, amount: float = 1.0) -> None:
        """Deposita ``amount`` en la celda (row, col) del campo local."""
        h, w = self.presence_field.shape
        if 0 <= row < h and 0 <= col < w:
            self.presence_field[row, col] += float(amount)

    def decay_presence(
        self,
        evaporation_rate: float = 0.05,
        diffusion_sigma: float = 0.0,
        diffuse_now: bool = False,
    ) -> None:
        """Evapora (y opcionalmente difunde) el campo local de presencia."""
        if evaporation_rate > 0:
            self.presence_field *= (1.0 - float(evaporation_rate))
        if diffuse_now and diffusion_sigma > 0:
            try:
                from scipy.ndimage import gaussian_filter as _gf
            except ImportError:
                return
            self.presence_field = _gf(
                self.presence_field,
                sigma=float(diffusion_sigma),
                mode="nearest",
            ).astype(np.float32)

    def merge_presence(self, other_field: np.ndarray) -> None:
        """Mezcla la feromona de otro agente vía max-merge (gossip).

        Semántica info-pesimista: si el peer sabía que una celda estaba
        más pisada, el receptor adopta ese valor. Tras el merge ambos
        evaporan independientemente.
        """
        if other_field.shape != self.presence_field.shape:
            return
        np.maximum(self.presence_field, other_field, out=self.presence_field)

    def get_updates_since(self, since_timestep: int) -> list[MapUpdate]:
        """Devuelve actualizaciones generadas desde since_timestep."""
        return [
            u for u in self._latest_updates.values()
            if u.timestamp >= since_timestep
        ]

    def merge_updates(self, incoming: list[MapUpdate], max_hops: int = 5) -> None:
        """Mezcla actualizaciones recibidas de otro agente (protocolo gossip).

        Las que han viajado más de max_hops saltos se descartan.
        """
        for update in incoming:
            if update.hops >= max_hops:
                continue

            key = (update.cell[0], update.cell[1], update.layer)
            existing = self._latest_updates.get(key)

            # Accept if no prior info or if the incoming update is newer
            if existing is None or update.timestamp > existing.timestamp:
                r, c = update.cell
                if update.layer == "exploration":
                    # Solo actualizar si el timestamp es más reciente
                    prev_ts = self.cells_gossip_explored.get((r, c), -1)
                    if update.timestamp > prev_ts:
                        self.cells_gossip_explored[(r, c)] = update.timestamp
                    self.exploration_map[r, c] = max(
                        self.exploration_map[r, c], update.value
                    )
                elif update.layer == "alert":
                    self.alert_map[r, c] = max(
                        self.alert_map[r, c], update.value
                    )

                # Incrementar hops antes de re-propagar
                relayed = MapUpdate(
                    cell=update.cell,
                    layer=update.layer,
                    value=update.value,
                    timestamp=update.timestamp,
                    origin_agent=update.origin_agent,
                    hops=update.hops + 1,
                )
                self._latest_updates[key] = relayed

        # Podar si el buffer ha crecido demasiado (limpiar entradas antiguas)
        self._prune_updates()

    def _prune_updates(self) -> None:
        """Elimina las entradas más antiguas si se supera el límite del buffer."""
        if len(self._latest_updates) <= _UPDATE_BUFFER_LIMIT:
            return
        # Ordenar por timestamp y quedarnos con las más recientes
        sorted_keys = sorted(
            self._latest_updates,
            key=lambda k: self._latest_updates[k].timestamp,
        )
        to_remove = len(sorted_keys) - _UPDATE_BUFFER_LIMIT
        for key in sorted_keys[:to_remove]:
            del self._latest_updates[key]

    # -- Evaporación de feromonas --

    def evaporate(
        self,
        exploration_rate: float = 0.01,
        alert_rate: float = 0.005,
    ) -> None:
        """Decae ambas capas de feromonas. Se llama una vez por tick."""
        self.exploration_map *= 1.0 - exploration_rate
        self.alert_map *= 1.0 - alert_rate
