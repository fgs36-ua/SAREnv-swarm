# sarenv/swarm/comparative.py
"""
Evaluador comparativo que ejecuta la simulación de enjambre y los
algoritmos centralizados (greedy, spiral, pizza, etc.) sobre el MISMO
escenario, y devuelve un DataFrame unificado con todas las métricas.

Es el corazón de la comparativa experimental del TFG.
"""
from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from shapely.geometry import LineString

from sarenv.analytics.metrics import PathEvaluator
from sarenv.analytics import paths as path_algorithms
from sarenv.core.loading import DatasetLoader, SARDatasetItem
from sarenv.core.lost_person import LostPersonLocationGenerator
from sarenv.utils.logging_setup import get_logger

from .config import SwarmConfig, DroneConfig, RobotDogConfig
from .metrics import SwarmMetrics
from .simulator import SwarmSimulator

log = get_logger()

# Claves comunes a todas las filas del DataFrame de comparativa.
# Centralizadas para evitar drift entre _evaluate_swarm/_evaluate_baseline/_empty_row.
METRIC_KEYS: tuple[str, ...] = (
    "n_agents", "num_drones", "num_dogs", "max_hops", "Budget_m",
    "Coverage_ratio", "Prob_covered_ratio", "Overlap_ratio",
    "Victims_pct", "Area_km2", "Path_length_km",
    "Likelihood", "TD_Score",
    "Time_first_victim", "Efficiency_ratio", "Latency", "Elapsed_s",
)


class SwarmComparativeEvaluator:
    """Ejecuta el enjambre y los algoritmos centralizados en exactamente
    los mismos escenarios (misma zona, mismas víctimas, mismo budget)
    y devuelve las métricas comparativas en un DataFrame.

    Parámetros de uso típico::

        evaluator = SwarmComparativeEvaluator(
            dataset_dir="maigmo_dataset",
            size="medium",
            num_victims=200,
            seeds=[42, 123, 456],
        )
        df = evaluator.run_all()
        df.to_csv("results/comparativa.csv", index=False)
    """

    # Algoritmos centralizados a incluir en la comparativa
    BASELINE_ALGORITHMS = {
        "Greedy": path_algorithms.generate_greedy_path,
        "Spiral": path_algorithms.generate_spiral_path,
        "Pizza": path_algorithms.generate_pizza_zigzag_path,
    }

    def __init__(
        self,
        dataset_dir: str = "maigmo_dataset",
        size: str = "medium",
        num_victims: int = 200,
        seeds: list[int] | None = None,
        # Parámetros del enjambre
        swarm_configs: list[dict[str, Any]] | None = None,
        # Parámetros compartidos
        budget_per_agent: float = 100_000.0,
        fov_deg: float = 45.0,
        altitude: float = 80.0,
        discount_factor: float = 0.999,
        anti_revisit_window: int = 4,
        anti_revisit_penalty: float = 0.05,
        presence_weight: float = 0.05,
        presence_diffusion_sigma: float = 0.5,
        pheromone_attenuation: float = 0.1,
        dispersal_weight: float = 0.1,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.size = size
        self.num_victims = num_victims
        self.seeds = seeds or [42, 123, 456]
        self.budget_per_agent = budget_per_agent
        self.fov_deg = fov_deg
        self.altitude = altitude
        self.discount_factor = discount_factor
        self.anti_revisit_window = anti_revisit_window
        self.anti_revisit_penalty = anti_revisit_penalty
        self.presence_weight = presence_weight
        self.presence_diffusion_sigma = presence_diffusion_sigma
        self.pheromone_attenuation = pheromone_attenuation
        self.dispersal_weight = dispersal_weight

        # Configuraciones de enjambre a probar (cada una es un escenario)
        # Si no se pasan, se usa una configuración por defecto
        self.swarm_configs = swarm_configs or [
            {"num_drones": 5, "num_dogs": 0, "max_hops": 3, "label": "Swarm_5D"},
        ]

    def run_all(self) -> pd.DataFrame:
        """Ejecuta todos los experimentos y devuelve un DataFrame unificado.

        Columnas: Seed, Algorithm, n_agents, Budget_m, Coverage_ratio,
        Prob_covered_ratio, Overlap_ratio, Victims_pct, Area_km2,
        Path_length_km, Likelihood, TD_Score, Time_first_victim,
        Efficiency_ratio, Latency, Elapsed_s
        """
        rows: list[dict] = []

        for seed in self.seeds:
            log.info(f"=== Seed {seed} ===")

            # 1. Cargar entorno y generar víctimas (mismas para todos)
            item, victims_gdf = self._load_scenario(seed)
            if item is None:
                continue

            # 2. Evaluar cada configuración de enjambre
            for cfg_dict in self.swarm_configs:
                label = cfg_dict.get("label", "Swarm")
                row = self._evaluate_swarm(item, victims_gdf, cfg_dict, seed)
                row["Algorithm"] = label
                row["Seed"] = seed
                rows.append(row)

            # 3. Evaluar algoritmos centralizados (baselines)
            n_agents = self.swarm_configs[0].get("num_drones", 5) + \
                       self.swarm_configs[0].get("num_dogs", 0)
            for algo_name, algo_func in self.BASELINE_ALGORITHMS.items():
                row = self._evaluate_baseline(
                    item, victims_gdf, algo_name, algo_func, n_agents, seed
                )
                row["Algorithm"] = algo_name
                row["Seed"] = seed
                rows.append(row)

        df = pd.DataFrame(rows)
        return df

    # ── Escenario ──────────────────────────────────────────────────

    def _load_scenario(self, seed: int):
        """Carga dataset y genera víctimas con la semilla dada."""
        import random
        random.seed(seed)
        np.random.seed(seed)

        try:
            loader = DatasetLoader(dataset_directory=self.dataset_dir)
            item = loader.load_environment(self.size)
        except Exception as e:
            log.error(f"Error cargando dataset: {e}")
            return None, None

        if item is None:
            log.error("No se pudo cargar el dataset.")
            return None, None

        victim_gen = LostPersonLocationGenerator(item)
        victim_points = victim_gen.generate_locations(
            self.num_victims, percent_random_samples=0
        )
        import geopandas as gpd
        data_crs = victim_gen.features.crs
        victims_gdf = (
            gpd.GeoDataFrame(geometry=victim_points, crs=data_crs)
            if victim_points
            else gpd.GeoDataFrame(columns=["geometry"], crs=data_crs)
        )
        log.info(f"  Escenario: {item.heatmap.shape}, {len(victims_gdf)} víctimas")
        return item, victims_gdf

    # ── Enjambre ───────────────────────────────────────────────────

    def _evaluate_swarm(
        self, item: SARDatasetItem, victims_gdf, cfg_dict: dict, seed: int
    ) -> dict:
        """Ejecuta una simulación de enjambre y extrae métricas."""
        num_drones = cfg_dict.get("num_drones", 5)
        num_dogs = cfg_dict.get("num_dogs", 0)
        max_hops = cfg_dict.get("max_hops", 3)
        max_steps = cfg_dict.get("max_steps", 15_000)

        drone_cfg = DroneConfig(altitude=self.altitude, fov_deg=self.fov_deg)
        dog_cfg = RobotDogConfig(sensor_range=20.0)
        for c in (drone_cfg, dog_cfg):
            c.anti_revisit_window = self.anti_revisit_window
            c.anti_revisit_penalty = self.anti_revisit_penalty
            c.presence_weight = self.presence_weight
            c.pheromone_attenuation = self.pheromone_attenuation
            c.dispersal_weight = self.dispersal_weight

        config = SwarmConfig(
            num_drones=num_drones,
            num_dogs=num_dogs,
            budget_per_agent=self.budget_per_agent,
            max_steps=max_steps,
            max_hops=max_hops,
            drone_config=drone_cfg,
            dog_config=dog_cfg,
            presence_diffusion_sigma=self.presence_diffusion_sigma,
        )

        t0 = time.perf_counter()
        sim = SwarmSimulator.from_dataset_item(item, config, seed=seed)
        sim.run()
        elapsed = time.perf_counter() - t0

        metrics = SwarmMetrics(sim, victims=victims_gdf, discount_factor=self.discount_factor)
        report = metrics.full_report()

        # PathEvaluator para métricas geoespaciales (likelihood, area, victims %)
        pe_metrics = metrics.evaluate_with_path_evaluator()

        n_agents = num_drones + num_dogs
        victim_pct = pe_metrics["victim_detection_metrics"].get("percentage_found", 0)

        return {
            "n_agents": n_agents,
            "num_drones": num_drones,
            "num_dogs": num_dogs,
            "max_hops": max_hops,
            "Budget_m": self.budget_per_agent,
            "Coverage_ratio": report["coverage_ratio"],
            "Prob_covered_ratio": report["probability_coverage_ratio"],
            "Overlap_ratio": report["overlap_ratio"],
            "Victims_pct": victim_pct,
            "Area_km2": pe_metrics["area_covered"],
            "Path_length_km": pe_metrics["total_path_length"],
            "Likelihood": pe_metrics["total_likelihood_score"],
            "TD_Score": pe_metrics["total_time_discounted_score"],
            "Time_first_victim": report.get("time_to_first_victim"),
            "Efficiency_ratio": report["efficiency_ratio"],
            "Latency": report["info_propagation_latency"],
            "Elapsed_s": round(elapsed, 1),
        }

    def _evaluate_swarm_with_failures(
        self, item: SARDatasetItem, victims_gdf, cfg_dict: dict,
        seed: int, *, kill_fraction: float = 0.0, kill_at_step: int = 2000,
    ) -> dict:
        """Ejecuta enjambre matando *kill_fraction* agentes en *kill_at_step*.

        Permite medir resiliencia: si kill_fraction=0.4 y hay 5 agentes,
        se desactivan 2 tras 2000 ticks.
        """
        num_drones = cfg_dict.get("num_drones", 5)
        num_dogs = cfg_dict.get("num_dogs", 0)
        max_hops = cfg_dict.get("max_hops", 1)
        max_steps = cfg_dict.get("max_steps", 15_000)

        drone_cfg = DroneConfig(altitude=self.altitude, fov_deg=self.fov_deg)
        dog_cfg = RobotDogConfig(sensor_range=20.0)
        for c in (drone_cfg, dog_cfg):
            c.anti_revisit_window = self.anti_revisit_window
            c.anti_revisit_penalty = self.anti_revisit_penalty
            c.presence_weight = self.presence_weight
            c.pheromone_attenuation = self.pheromone_attenuation
            c.dispersal_weight = self.dispersal_weight

        config = SwarmConfig(
            num_drones=num_drones,
            num_dogs=num_dogs,
            budget_per_agent=self.budget_per_agent,
            max_steps=max_steps,
            max_hops=max_hops,
            drone_config=drone_cfg,
            dog_config=dog_cfg,
            presence_diffusion_sigma=self.presence_diffusion_sigma,
        )

        t0 = time.perf_counter()
        sim = SwarmSimulator.from_dataset_item(item, config, seed=seed)

        n_agents = num_drones + num_dogs
        n_kill = int(round(kill_fraction * n_agents))
        killed = False

        for step_i in range(max_steps):
            if not killed and step_i >= kill_at_step and n_kill > 0:
                active_ids = [a.id for a in sim.agents if a.active]
                for aid in active_ids[:n_kill]:
                    sim.kill_agent(aid)
                killed = True
            sim.step()
            if not any(a.active for a in sim.agents):
                break

        elapsed = time.perf_counter() - t0

        metrics = SwarmMetrics(sim, victims=victims_gdf, discount_factor=self.discount_factor)
        report = metrics.full_report()
        pe_metrics = metrics.evaluate_with_path_evaluator()
        victim_pct = pe_metrics["victim_detection_metrics"].get("percentage_found", 0)

        return {
            "n_agents": n_agents,
            "num_drones": num_drones,
            "num_dogs": num_dogs,
            "max_hops": max_hops,
            "kill_fraction": kill_fraction,
            "Budget_m": self.budget_per_agent,
            "Coverage_ratio": report["coverage_ratio"],
            "Prob_covered_ratio": report["probability_coverage_ratio"],
            "Overlap_ratio": report["overlap_ratio"],
            "Victims_pct": victim_pct,
            "Area_km2": pe_metrics["area_covered"],
            "Path_length_km": pe_metrics["total_path_length"],
            "Likelihood": pe_metrics["total_likelihood_score"],
            "TD_Score": pe_metrics["total_time_discounted_score"],
            "Time_first_victim": report.get("time_to_first_victim"),
            "Efficiency_ratio": report["efficiency_ratio"],
            "Latency": report["info_propagation_latency"],
            "Elapsed_s": round(elapsed, 1),
        }

    # ── Algoritmos centralizados ───────────────────────────────────

    def _make_path_evaluator(self, item: SARDatasetItem, victims_gdf) -> PathEvaluator:
        """Construye un PathEvaluator con los parámetros estándar del evaluador."""
        meters_per_bin = int(np.ceil(
            (item.bounds[2] - item.bounds[0]) / item.heatmap.shape[1]
        ))
        return PathEvaluator(
            heatmap=item.heatmap,
            extent=item.bounds,
            victims=victims_gdf,
            fov_deg=self.fov_deg,
            altitude=self.altitude,
            meters_per_bin=meters_per_bin,
        )

    def _evaluate_baseline(
        self, item: SARDatasetItem, victims_gdf,
        algo_name: str, algo_func, n_agents: int, seed: int,
    ) -> dict:
        """Ejecuta un algoritmo centralizado y extrae métricas comparables."""
        env = item
        center_x = (env.bounds[0] + env.bounds[2]) / 2
        center_y = (env.bounds[1] + env.bounds[3]) / 2
        max_radius = max(
            env.bounds[2] - env.bounds[0],
            env.bounds[3] - env.bounds[1],
        ) / 2

        t0 = time.perf_counter()
        # Los baselines interpretan 'budget' como budget TOTAL y lo dividen
        # entre num_drones internamente.  Para comparar justamente,
        # el total del baseline debe ser = budget_per_agent × n_agentes.
        total_budget = self.budget_per_agent * n_agents
        try:
            paths_result = algo_func(
                center_x=center_x,
                center_y=center_y,
                max_radius=max_radius,
                num_drones=n_agents,
                fov_deg=self.fov_deg,
                altitude=self.altitude,
                overlap=0.0,
                path_point_spacing_m=10.0,
                budget=total_budget,
                probability_map=env.heatmap,
                bounds=env.bounds,
                border_gap_m=15.0,
                transition_distance_m=50.0,
            )
        except Exception as e:
            log.warning(f"Error en {algo_name}: {e}")
            paths_result = []
        elapsed = time.perf_counter() - t0

        if not paths_result:
            return self._empty_row(n_agents, elapsed)

        pe = self._make_path_evaluator(env, victims_gdf)
        pe_metrics = pe.calculate_all_metrics(paths_result, self.discount_factor)
        victim_pct = pe_metrics["victim_detection_metrics"].get("percentage_found", 0)

        row = dict.fromkeys(METRIC_KEYS)
        row.update({
            "n_agents": n_agents,
            "num_drones": n_agents,
            "num_dogs": 0,
            "Budget_m": self.budget_per_agent,
            "Victims_pct": victim_pct,
            "Area_km2": pe_metrics["area_covered"],
            "Path_length_km": pe_metrics["total_path_length"],
            "Likelihood": pe_metrics["total_likelihood_score"],
            "TD_Score": pe_metrics["total_time_discounted_score"],
            "Elapsed_s": round(elapsed, 1),
        })
        return row

    def _empty_row(self, n_agents: int, elapsed: float) -> dict:
        row = dict.fromkeys(METRIC_KEYS)
        row.update({
            "n_agents": n_agents, "num_drones": n_agents, "num_dogs": 0,
            "Budget_m": self.budget_per_agent,
            "Victims_pct": 0, "Area_km2": 0, "Path_length_km": 0,
            "Likelihood": 0, "TD_Score": 0,
            "Elapsed_s": round(elapsed, 1),
        })
        return row

    def _evaluate_baseline_with_failures(
        self, item: SARDatasetItem, victims_gdf,
        algo_name: str, algo_func, n_agents: int, seed: int,
        *, kill_fraction: float = 0.0, path_fraction_before_kill: float = 0.3,
    ) -> dict:
        """Evalúa un baseline simulando pérdida de agentes.

        En un sistema centralizado, si un dron falla a mitad de ruta, su
        sector queda SIN CUBRIR porque nadie re-planifica.  Simulamos esto
        truncando *kill_fraction* caminos al *path_fraction_before_kill*
        de su longitud total.
        """
        env = item
        center_x = (env.bounds[0] + env.bounds[2]) / 2
        center_y = (env.bounds[1] + env.bounds[3]) / 2
        max_radius = max(
            env.bounds[2] - env.bounds[0],
            env.bounds[3] - env.bounds[1],
        ) / 2

        total_budget = self.budget_per_agent * n_agents
        t0 = time.perf_counter()
        try:
            paths_result = algo_func(
                center_x=center_x, center_y=center_y,
                max_radius=max_radius, num_drones=n_agents,
                fov_deg=self.fov_deg, altitude=self.altitude,
                overlap=0.0, path_point_spacing_m=10.0,
                budget=total_budget, probability_map=env.heatmap,
                bounds=env.bounds, border_gap_m=15.0,
                transition_distance_m=50.0,
            )
        except Exception as e:
            log.warning(f"Error en {algo_name}: {e}")
            paths_result = []
        elapsed = time.perf_counter() - t0

        if not paths_result:
            return self._empty_row(n_agents, elapsed)

        # Truncar paths de agentes "muertos"
        n_kill = int(round(kill_fraction * len(paths_result)))
        for i in range(n_kill):
            path = paths_result[i]
            if path.is_empty or path.length == 0:
                continue
            cut_length = path.length * path_fraction_before_kill
            truncated = path.interpolate(cut_length)
            # Reconstruir LineString hasta el punto de corte
            coords = list(path.coords)
            new_coords = []
            acc = 0.0
            new_coords.append(coords[0])
            for j in range(1, len(coords)):
                seg = LineString([coords[j - 1], coords[j]])
                if acc + seg.length >= cut_length:
                    # Interpolar punto exacto
                    new_coords.append(truncated.coords[0])
                    break
                acc += seg.length
                new_coords.append(coords[j])
            if len(new_coords) >= 2:
                paths_result[i] = LineString(new_coords)
            else:
                paths_result[i] = LineString()

        pe = self._make_path_evaluator(env, victims_gdf)
        pe_metrics = pe.calculate_all_metrics(paths_result, self.discount_factor)
        victim_pct = pe_metrics["victim_detection_metrics"].get("percentage_found", 0)

        row = dict.fromkeys(METRIC_KEYS)
        row.update({
            "n_agents": n_agents, "num_drones": n_agents, "num_dogs": 0,
            "Budget_m": self.budget_per_agent,
            "Victims_pct": victim_pct,
            "Area_km2": pe_metrics["area_covered"],
            "Path_length_km": pe_metrics["total_path_length"],
            "Likelihood": pe_metrics["total_likelihood_score"],
            "TD_Score": pe_metrics["total_time_discounted_score"],
            "Elapsed_s": round(elapsed, 1),
        })
        row["kill_fraction"] = kill_fraction
        return row
