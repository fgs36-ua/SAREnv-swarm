# sarenv/swarm/metrics.py
"""
Adaptador que evalúa los resultados de la simulación de enjambre usando
el PathEvaluator de sarenv.analytics.metrics, más métricas específicas
del enjambre (cobertura, solapamiento, contribución por agente, tiempo
hasta primera víctima, latencia de propagación y aglomeración).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from shapely.geometry import LineString, Point

if TYPE_CHECKING:
    import geopandas as gpd
    from .simulator import SwarmSimulator


class SwarmMetrics:
    """Evaluación de una simulación de enjambre completada.

    Envuelve PathEvaluator y añade métricas propias del enjambre:
    solapamiento de exploración, contribución por agente, etc.
    """

    def __init__(
        self,
        simulator: SwarmSimulator,
        victims: gpd.GeoDataFrame | None = None,
        discount_factor: float = 0.999,
    ) -> None:
        self.sim = simulator
        self.victims = victims
        self.discount_factor = discount_factor

    # -- Puente con PathEvaluator --

    def evaluate_with_path_evaluator(
        self,
        fov_deg: float | None = None,
        altitude: float | None = None,
        meters_per_bin: int | None = None,
    ) -> dict:
        """Ejecuta PathEvaluator.calculate_all_metrics sobre los paths del enjambre.

        Los parámetros que falten se derivan de la config del dron.
        """
        from sarenv.analytics.metrics import PathEvaluator

        env = self.sim.env
        cfg = self.sim.config.drone_config

        fov_deg = fov_deg or cfg.fov_deg
        altitude = altitude or cfg.altitude
        meters_per_bin = meters_per_bin or int(np.ceil(env.grid.dx))

        evaluator = PathEvaluator(
            heatmap=env.raw_heatmap,
            extent=env.bounds,
            victims=self._ensure_victims_gdf(),
            fov_deg=fov_deg,
            altitude=altitude,
            meters_per_bin=meters_per_bin,
        )

        paths = self.sim.get_paths()
        return evaluator.calculate_all_metrics(paths, self.discount_factor)

    # -- Métricas específicas del enjambre --

    def coverage_summary(self) -> dict:
        """Resumen rápido de cobertura usando los mapas de conocimiento
        de los agentes (sin necesidad de PathEvaluator).
        """
        env = self.sim.env
        total_cells = env.grid.rows * env.grid.cols

        # Usar cells_ever_explored (acumulativo, inmune a evaporación)
        # en vez del exploration_map que decae cada tick.
        per_agent_explored: dict[str, int] = {}
        all_explored: set[tuple[int, int]] = set()

        for agent in self.sim.agents:
            per_agent_explored[agent.id] = len(agent.cells_ever_explored)
            all_explored.update(agent.cells_ever_explored)

        total_explored = len(all_explored)

        # Solapamiento: celdas exploradas por más de un agente
        sum_individual = sum(per_agent_explored.values())
        overlap = (sum_individual - total_explored) if sum_individual > 0 else 0

        # Masa de probabilidad cubierta (sobre el heatmap ORIGINAL, sin normalizar)
        prob_covered = 0.0
        for r, c in all_explored:
            prob_covered += float(env.raw_heatmap[r, c])
        prob_total = float(np.sum(env.raw_heatmap))

        return {
            "total_cells": total_cells,
            "explored_cells": total_explored,
            "coverage_ratio": total_explored / total_cells if total_cells else 0,
            "overlap_cells": overlap,
            "overlap_ratio": overlap / sum_individual if sum_individual else 0,
            "probability_covered": prob_covered,
            "probability_total": prob_total,
            "probability_coverage_ratio": (
                prob_covered / prob_total if prob_total > 0 else 0
            ),
            "per_agent_explored": per_agent_explored,
            "total_timesteps": self.sim.timestep,
            "paths_lengths_m": {
                a.id: a.get_path_linestring().length for a in self.sim.agents
            },
            # Budget consumido (con coste de terreno) vs distancia geométrica
            "budget_consumed_m": {
                a.id: a.config.budget - a.budget_remaining
                for a in self.sim.agents
            },
        }

    # -- Métricas extendidas (tiempo, latencia, eficiencia) --

    def time_to_first_victim(self) -> int | None:
        """Tick en que algún agente detectó la primera víctima.

        Calcula la distancia entre cada víctima y la posición de cada agente
        en cada tick (usando el historial de posiciones). Se considera
        detectada si la víctima cae dentro del radio de detección del agente.

        Returns:
            El tick (int) o None si no se detectó ninguna víctima.
        """
        import geopandas as gpd
        victims_gdf = self._ensure_victims_gdf()
        if victims_gdf.empty:
            return None

        env = self.sim.env
        victim_cells: list[tuple[int, int]] = []
        for geom in victims_gdf.geometry:
            r, c = env.world_to_grid(geom.x, geom.y)
            r = int(np.clip(r, 0, env.grid.rows - 1))
            c = int(np.clip(c, 0, env.grid.cols - 1))
            victim_cells.append((r, c))

        victim_set = set(victim_cells)

        # Recorrer el historial de posiciones (agent.path) de cada agente
        # y comprobar si en algún tick cubre alguna víctima.
        # Encontrar el tick más temprano entre todos los agentes.
        earliest_tick = None
        for agent in self.sim.agents:
            radius_cells = max(1, int(agent.config.detection_radius / env.grid.dx))
            for tick, pos in enumerate(agent.path):
                r, c = pos
                for dr in range(-radius_cells, radius_cells + 1):
                    for dc in range(-radius_cells, radius_cells + 1):
                        nr, nc = r + dr, c + dc
                        if (nr, nc) in victim_set:
                            if earliest_tick is None or tick < earliest_tick:
                                earliest_tick = tick
                            # No necesitamos seguir con este agente
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break  # ya encontramos el primer tick de este agente
        return earliest_tick

    def effective_vs_redundant_coverage(self) -> dict:
        """Descompone la exploración total en cobertura efectiva (nueva)
        y redundante (celdas ya visitadas por otro agente).

        Returns:
            dict con 'effective_cells', 'redundant_visits', 'efficiency_ratio'
        """
        all_explored: set[tuple[int, int]] = set()
        total_visits = 0

        for agent in self.sim.agents:
            total_visits += len(agent.cells_ever_explored)
            all_explored.update(agent.cells_ever_explored)

        effective = len(all_explored)
        redundant = total_visits - effective
        efficiency = effective / total_visits if total_visits > 0 else 0

        return {
            "effective_cells": effective,
            "redundant_visits": redundant,
            "total_visits": total_visits,
            "efficiency_ratio": efficiency,
        }

    def information_propagation_latency(self) -> float:
        """Latencia media de propagación de información entre agentes.

        Mide cuántos ticks tarda la información gossip en llegar de un agente
        a otro, usando los timestamps de las celdas en cells_gossip_explored.
        Compara el timestamp del gossip recibido con el timestamp actual de la
        simulación para estimar el retraso medio.

        Returns:
            Latencia media en ticks (float). 0 si no hay datos gossip.
        """
        total_latency = 0
        count = 0

        for agent in self.sim.agents:
            gossip = agent.knowledge.cells_gossip_explored
            if not gossip:
                continue
            for _cell, timestamp in gossip.items():
                # El timestamp guardado es cuándo se recibió; la info original
                # fue generada antes. La diferencia con el tick final da una
                # estimación pobre, así que usamos el spread: cuánto tardó
                # en llegar respecto al momento en que se generó.
                # Como aproximación: latency = sim.timestep - timestamp
                latency = self.sim.timestep - timestamp
                total_latency += latency
                count += 1

        return total_latency / count if count > 0 else 0.0

    def full_report(self) -> dict:
        """Informe completo combinando todas las métricas del enjambre.

        Combina coverage_summary con las métricas extendidas
        en un solo diccionario para facilitar la comparativa.
        """
        summary = self.coverage_summary()
        eff = self.effective_vs_redundant_coverage()
        ttfv = self.time_to_first_victim()
        latency = self.information_propagation_latency()

        summary["effective_cells"] = eff["effective_cells"]
        summary["redundant_visits"] = eff["redundant_visits"]
        summary["efficiency_ratio"] = eff["efficiency_ratio"]
        summary["time_to_first_victim"] = ttfv
        summary["info_propagation_latency"] = latency

        # Métricas de aglomeración (ver docs/16)
        agg = self.aggregation_report()
        summary["coverage_gini"] = agg["coverage_gini"]
        summary["cluster_ratio"] = agg["cluster_ratio"]
        summary["mean_pair_distance_cells"] = agg["mean_pair_distance_cells"]

        return summary

    # ------------------------------------------------------------------
    # Métricas de aglomeración / dispersión espacial
    #
    # Referencias:
    # - Coverage Gini coefficient: Gini, C. (1912), "Variabilità e
    #   mutabilità". Aplicado en swarm robotics como métrica de equidad
    #   de la distribución de visitas (Hsieh et al. 2008).
    # - Tiempo en clúster: Reynolds (1987) "Flocks, herds, and schools"
    #   define la regla de "separation"; el tiempo medio que un agente
    #   pasa con vecinos cercanos cuantifica su violación.
    # - Distancia media entre pares: Cortés et al. (2004) "Coverage
    #   control for mobile sensing networks", como proxy de cobertura
    #   uniforme tipo Voronoi.
    # ------------------------------------------------------------------

    def coverage_gini(self) -> float:
        """Gini sobre el número de visitas por celda visitada.

        - Gini = 0  → todas las celdas visitadas reciben el mismo número
          de visitas (cobertura perfectamente uniforme).
        - Gini → 1  → unas pocas celdas concentran casi todas las visitas
          (aglomeración fuerte).

        Returns
        -------
        float
            Gini en [0, 1]. 0.0 si no hay datos.
        """
        env = self.sim.env
        visits = np.zeros((env.grid.rows, env.grid.cols), dtype=np.int32)
        for agent in self.sim.agents:
            for r, c in agent.path:
                visits[r, c] += 1

        flat = visits[visits > 0]
        if flat.size == 0:
            return 0.0
        sorted_v = np.sort(flat).astype(np.float64)
        n = sorted_v.size
        total = sorted_v.sum()
        if total <= 0:
            return 0.0
        gini = (2.0 * np.sum(np.arange(1, n + 1) * sorted_v)) / (n * total)
        gini -= (n + 1.0) / n
        return float(gini)

    def time_in_cluster(
        self,
        radius_cells: int = 5,
        min_neighbors: int = 2,
    ) -> dict:
        """Cuantifica la aglomeración tick-a-tick.

        Para cada tick, cuenta cuántos agentes tienen al menos
        ``min_neighbors`` otros agentes dentro de ``radius_cells`` celdas
        (distancia Chebyshev). Devuelve la fracción de pares
        (agente, tick) en esa situación, además de la distancia media
        por pares promediada en el tiempo.

        Returns
        -------
        dict
            ``cluster_ratio`` ∈ [0, 1], ``mean_pair_distance_cells``,
            ``ticks_evaluated``, ``radius_cells``, ``min_neighbors``.
        """
        agents = self.sim.agents
        empty = {
            "cluster_ratio": 0.0,
            "mean_pair_distance_cells": 0.0,
            "ticks_evaluated": 0,
            "radius_cells": radius_cells,
            "min_neighbors": min_neighbors,
        }
        if len(agents) < 2:
            return empty
        min_len = min(len(a.path) for a in agents)
        if min_len < 1:
            return empty

        # Apilar paths en (N, T, 2). Truncamos al min común para evitar
        # ragged arrays cuando algún agente terminó antes (return-to-base).
        positions = np.array(
            [[a.path[t] for t in range(min_len)] for a in agents],
            dtype=np.int32,
        )

        n = len(agents)
        cluster_count = 0
        total_evaluations = 0
        pair_dist_sum = 0.0
        pair_dist_count = 0
        iu = np.triu_indices(n, k=1)

        for t in range(min_len):
            pos_t = positions[:, t, :]  # (N, 2)
            diff = np.abs(pos_t[:, None, :] - pos_t[None, :, :])
            dist = diff.max(axis=2)  # Chebyshev
            neighbors_within = (dist <= radius_cells).sum(axis=1) - 1
            cluster_count += int((neighbors_within >= min_neighbors).sum())
            total_evaluations += n
            if iu[0].size > 0:
                pair_dist_sum += float(dist[iu].sum())
                pair_dist_count += iu[0].size

        return {
            "cluster_ratio": (
                cluster_count / total_evaluations if total_evaluations else 0.0
            ),
            "mean_pair_distance_cells": (
                pair_dist_sum / pair_dist_count if pair_dist_count else 0.0
            ),
            "ticks_evaluated": min_len,
            "radius_cells": radius_cells,
            "min_neighbors": min_neighbors,
        }

    def aggregation_report(
        self,
        radius_cells: int = 5,
        min_neighbors: int = 2,
    ) -> dict:
        """Combina ``coverage_gini`` y ``time_in_cluster``."""
        gini = self.coverage_gini()
        cluster = self.time_in_cluster(
            radius_cells=radius_cells, min_neighbors=min_neighbors,
        )
        return {"coverage_gini": gini, **cluster}

    # -- Helpers privados --

    def _ensure_victims_gdf(self):
        """Devuelve un GeoDataFrame válido (posiblemente vacío) para PathEvaluator."""
        import geopandas as gpd
        if self.victims is None:
            return gpd.GeoDataFrame(geometry=[], crs=None)
        if isinstance(self.victims, gpd.GeoDataFrame):
            if not self.victims.empty:
                return self.victims
            return gpd.GeoDataFrame(geometry=[], crs=None)
        # Handle list of Points
        if isinstance(self.victims, list) and len(self.victims) > 0:
            return gpd.GeoDataFrame(geometry=self.victims, crs=None)
        return gpd.GeoDataFrame(geometry=[], crs=None)
