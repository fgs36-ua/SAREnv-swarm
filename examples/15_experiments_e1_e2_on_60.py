# examples/15_experiments_e1_e2_on_60.py
"""
Experimentos E1 (Enjambre vs baselines) y E2 (composición heterogénea)
sobre 8 escenarios SAREnv estratificados (2 por grupo de terreno/clima).

E1 — Enjambre vs baselines:
    Compara Swarm_3D2P frente a los baselines centralizados Greedy y Pizza
    sobre los 8 escenarios × 3 seeds.

E2 — Composición heterogénea:
    Compara tres composiciones del enjambre con N=5 agentes:
        Swarm_5D0P  (5 drones, 0 perros)
        Swarm_3D2P  (3 drones, 2 perros)
        Swarm_0D5P  (0 drones, 5 perros)
    sobre los 8 escenarios × 3 seeds.

Reutiliza utilidades de `12_evaluate_60_scenarios.py`
(load_scenario_item, compute_budget, get_scenario_group).

Salida:
    results/exp_60scen_e1.csv  — filas con métricas para E1
    results/exp_60scen_e2.csv  — filas con métricas para E2
    graphs/exp_60scen_e1.png   — boxplot Victims_pct por algoritmo y grupo
    graphs/exp_60scen_e2.png   — boxplot Victims_pct por composición y grupo

Uso:
    python examples/15_experiments_e1_e2_on_60.py
    python examples/15_experiments_e1_e2_on_60.py --quick      # 4 scen × 1 seed
    python examples/15_experiments_e1_e2_on_60.py --only e1    # solo E1
    python examples/15_experiments_e1_e2_on_60.py --only e2    # solo E2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reutilizamos las utilidades del script 12.
sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
_mod = import_module("12_evaluate_60_scenarios")
load_scenario_item = _mod.load_scenario_item
compute_budget = _mod.compute_budget
get_scenario_group = _mod.get_scenario_group

from sarenv.core.lost_person import LostPersonLocationGenerator
from sarenv.swarm.config import SwarmConfig, DroneConfig, RobotDogConfig
from sarenv.swarm.metrics import SwarmMetrics
from sarenv.swarm.simulator import SwarmSimulator
from sarenv.utils.logging_setup import get_logger

log = get_logger()

RESULTS_DIR = Path("results")
GRAPHS_DIR = Path("graphs")
SEED_LIST = [42, 123, 456]

# Muestra estratificada (2 por grupo, igual que script 13).
SCENARIO_SAMPLE = [1, 5, 16, 20, 31, 35, 46, 50]

# Composiciones para E2.
E2_COMPOSITIONS = [
    ("Swarm_5D0P", 5, 0),
    ("Swarm_3D2P", 3, 2),
    ("Swarm_0D5P", 0, 5),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Experimentos E1 y E2 sobre 8 escenarios SAREnv"
    )
    p.add_argument("--scenarios", type=int, nargs="*", default=None,
                   help=f"Escenarios a usar (default: {SCENARIO_SAMPLE})")
    p.add_argument("--seeds", type=int, default=3,
                   help="Número de semillas (default: 3)")
    p.add_argument("--max-steps", type=int, default=15_000)
    p.add_argument("--num-victims", type=int, default=200)
    p.add_argument("--budget-factor", type=float, default=1.0)
    p.add_argument("--only", choices=["e1", "e2", "both"], default="both",
                   help="Qué experimento ejecutar (default: both)")
    p.add_argument("--quick", action="store_true",
                   help="Modo rápido: 4 escenarios × 1 seed")
    return p.parse_args()


def _generate_victims(item, seed: int, num_victims: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        gen = LostPersonLocationGenerator(item)
        pts = gen.generate_locations(num_victims, percent_random_samples=0)
        return gpd.GeoDataFrame(geometry=pts, crs=item.features.crs)
    except Exception as e:
        log.warning(f"  generador víctimas falló: {e}")
        return gpd.GeoDataFrame(columns=["geometry"], crs=item.features.crs)


def run_swarm(item, scenario_id: int, num_drones: int, num_dogs: int,
              seed: int, args, label: str) -> dict:
    """Ejecuta el enjambre y devuelve fila de métricas."""
    victims_gdf = _generate_victims(item, seed, args.num_victims)
    budget = compute_budget(item, args.budget_factor)

    drone_cfg = DroneConfig(altitude=80.0, fov_deg=45.0)
    dog_cfg = RobotDogConfig(sensor_range=20.0)
    swarm_cfg = SwarmConfig(
        num_drones=num_drones,
        num_dogs=num_dogs,
        budget_per_agent=budget,
        max_steps=args.max_steps,
        drone_config=drone_cfg,
        dog_config=dog_cfg,
    )

    t0 = time.perf_counter()
    sim = SwarmSimulator.from_dataset_item(item, swarm_cfg, seed=seed)
    sim.run()
    elapsed = time.perf_counter() - t0

    metrics = SwarmMetrics(sim, victims=victims_gdf)
    report = metrics.full_report()
    try:
        pe = metrics.evaluate_with_path_evaluator()
        victim_pct = pe["victim_detection_metrics"].get("percentage_found", 0)
        area_km2 = pe["area_covered"]
        likelihood = pe["total_likelihood_score"]
        path_km = pe["total_path_length"]
    except Exception as e:
        log.warning(f"  PathEvaluator falló: {e}")
        victim_pct = area_km2 = likelihood = path_km = 0

    return {
        "Scenario": scenario_id,
        "Group": get_scenario_group(scenario_id),
        "Environment": item.environment_type,
        "Climate": item.environment_climate,
        "Radius_km": item.radius_km,
        "Budget_m": round(budget, 1),
        "Algorithm": label,
        "n_agents": num_drones + num_dogs,
        "Seed": seed,
        "Victims_pct": round(victim_pct, 2),
        "Area_km2": round(area_km2, 2),
        "Coverage_ratio": round(report["coverage_ratio"], 4),
        "Prob_covered_ratio": round(report["probability_coverage_ratio"], 4),
        "Overlap_ratio": round(report["overlap_ratio"], 4),
        "Efficiency_ratio": round(report["efficiency_ratio"], 4),
        "Likelihood": round(likelihood, 6),
        "Path_km": round(path_km, 2),
        "Elapsed_s": round(elapsed, 1),
    }


def run_baseline(item, scenario_id: int, algo_name: str, algo_func,
                 n_agents: int, seed: int, args) -> dict:
    """Ejecuta un baseline centralizado y devuelve fila de métricas."""
    from sarenv.analytics.metrics import PathEvaluator
    from sarenv.analytics.evaluator import PathGeneratorConfig, PathGenerator

    victims_gdf = _generate_victims(item, seed, args.num_victims)
    budget = compute_budget(item, args.budget_factor)
    center_x = (item.bounds[0] + item.bounds[2]) / 2
    center_y = (item.bounds[1] + item.bounds[3]) / 2
    max_radius = max(item.bounds[2] - item.bounds[0],
                     item.bounds[3] - item.bounds[1]) / 2
    meters_per_bin = int(np.ceil(
        (item.bounds[2] - item.bounds[0]) / item.heatmap.shape[1]
    ))

    cfg = PathGeneratorConfig(num_drones=n_agents, budget=budget)
    gen = PathGenerator(name=algo_name, func=algo_func, path_generator_config=cfg)

    t0 = time.perf_counter()
    try:
        paths = gen(
            center_x=center_x, center_y=center_y, max_radius=max_radius,
            probability_map=item.heatmap, bounds=item.bounds,
        )
        elapsed = time.perf_counter() - t0

        evaluator = PathEvaluator(
            heatmap=item.heatmap, extent=item.bounds, victims=victims_gdf,
            fov_deg=45.0, altitude=80.0, meters_per_bin=meters_per_bin,
        )
        pe = evaluator.calculate_all_metrics(paths, discount_factor=1.0)
        victim_pct = pe["victim_detection_metrics"].get("percentage_found", 0)
        area_km2 = pe["area_covered"]
        likelihood = pe["total_likelihood_score"]
        path_km = pe["total_path_length"]
    except Exception as e:
        log.warning(f"  Baseline {algo_name} escenario {scenario_id} falló: {e}")
        elapsed = 0
        victim_pct = area_km2 = likelihood = path_km = 0

    return {
        "Scenario": scenario_id,
        "Group": get_scenario_group(scenario_id),
        "Environment": item.environment_type,
        "Climate": item.environment_climate,
        "Radius_km": item.radius_km,
        "Budget_m": round(budget, 1),
        "Algorithm": algo_name,
        "n_agents": n_agents,
        "Seed": seed,
        "Victims_pct": round(victim_pct, 2),
        "Area_km2": round(area_km2, 2),
        "Coverage_ratio": 0,
        "Prob_covered_ratio": 0,
        "Overlap_ratio": 0,
        "Efficiency_ratio": 0,
        "Likelihood": round(likelihood, 6),
        "Path_km": round(path_km, 2),
        "Elapsed_s": round(elapsed, 1),
    }


def run_experiment_e1(scenarios, seeds, args) -> pd.DataFrame:
    """E1: Swarm_3D2P vs Greedy vs Pizza."""
    from sarenv.analytics import paths as path_algorithms

    baselines = {
        "Greedy": path_algorithms.generate_greedy_path,
        "Pizza": path_algorithms.generate_pizza_zigzag_path,
    }
    rows: list[dict] = []
    total = len(scenarios) * len(seeds) * 3  # 1 swarm + 2 baselines
    i = 0
    t_start = time.perf_counter()
    out_partial = RESULTS_DIR / "exp_60scen_e1_partial.csv"

    log.info(f"E1: {total} simulaciones planificadas")
    for sid in scenarios:
        item = load_scenario_item(sid)
        if item is None:
            log.error(f"E1 escenario {sid}: imposible cargar, saltando")
            continue
        for seed in seeds:
            # Swarm 3D+2P
            i += 1
            elapsed_min = (time.perf_counter() - t_start) / 60.0
            log.info(f"  [E1 {i}/{total}] sid={sid} seed={seed} Swarm_3D2P "
                     f"(elapsed {elapsed_min:.1f} min)")
            rows.append(run_swarm(item, sid, 3, 2, seed, args, "Swarm_3D2P"))
            pd.DataFrame(rows).to_csv(out_partial, index=False)
            # Baselines
            for algo_name, algo_func in baselines.items():
                i += 1
                elapsed_min = (time.perf_counter() - t_start) / 60.0
                log.info(f"  [E1 {i}/{total}] sid={sid} seed={seed} {algo_name} "
                         f"(elapsed {elapsed_min:.1f} min)")
                rows.append(run_baseline(item, sid, algo_name, algo_func,
                                          5, seed, args))
                pd.DataFrame(rows).to_csv(out_partial, index=False)

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "exp_60scen_e1.csv"
    df.to_csv(out_csv, index=False)
    log.info(f"  >> E1 CSV: {out_csv} ({len(df)} filas)")
    return df


def run_experiment_e2(scenarios, seeds, args) -> pd.DataFrame:
    """E2: Swarm_5D0P vs Swarm_3D2P vs Swarm_0D5P."""
    rows: list[dict] = []
    total = len(scenarios) * len(seeds) * len(E2_COMPOSITIONS)
    i = 0
    t_start = time.perf_counter()
    out_partial = RESULTS_DIR / "exp_60scen_e2_partial.csv"

    log.info(f"E2: {total} simulaciones planificadas")
    for sid in scenarios:
        item = load_scenario_item(sid)
        if item is None:
            log.error(f"E2 escenario {sid}: imposible cargar, saltando")
            continue
        for seed in seeds:
            for label, nd, np_dogs in E2_COMPOSITIONS:
                i += 1
                elapsed_min = (time.perf_counter() - t_start) / 60.0
                log.info(f"  [E2 {i}/{total}] sid={sid} seed={seed} {label} "
                         f"(elapsed {elapsed_min:.1f} min)")
                rows.append(run_swarm(item, sid, nd, np_dogs, seed, args, label))
                pd.DataFrame(rows).to_csv(out_partial, index=False)

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "exp_60scen_e2.csv"
    df.to_csv(out_csv, index=False)
    log.info(f"  >> E2 CSV: {out_csv} ({len(df)} filas)")
    return df


def plot_boxplot(df: pd.DataFrame, out_png: Path, title: str) -> None:
    """Genera boxplot de Victims_pct por (Algorithm, Group)."""
    if df.empty:
        return
    groups = sorted(df["Group"].unique())
    algos = sorted(df["Algorithm"].unique())
    fig, axes = plt.subplots(1, len(groups), figsize=(4 * len(groups), 5),
                              sharey=True)
    if len(groups) == 1:
        axes = [axes]
    for ax, g in zip(axes, groups):
        sub = df[df["Group"] == g]
        data = [sub[sub["Algorithm"] == a]["Victims_pct"].values for a in algos]
        ax.boxplot(data, labels=algos)
        ax.set_title(g)
        ax.set_ylabel("Victims found (%)")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    log.info(f"  >> Gráfica: {out_png}")


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    if args.quick:
        scenarios = [1, 16, 31, 46]
        seeds = SEED_LIST[:1]
    else:
        scenarios = args.scenarios or SCENARIO_SAMPLE
        seeds = SEED_LIST[: args.seeds]

    t_global = time.perf_counter()
    log.info(f"Experimentos E1/E2 sobre {len(scenarios)} escenarios × "
             f"{len(seeds)} seeds. Mode: {args.only}")
    log.info(f"  Escenarios: {scenarios}")
    log.info(f"  Seeds: {seeds}")

    if args.only in ("e1", "both"):
        df_e1 = run_experiment_e1(scenarios, seeds, args)
        plot_boxplot(df_e1, GRAPHS_DIR / "exp_60scen_e1.png",
                     "E1 — Swarm_3D2P vs Greedy vs Pizza (8 SAREnv scenarios)")
        log.info("\n" + "=" * 70)
        log.info("  RESUMEN E1 — Victims_pct (media ± std)")
        log.info("=" * 70)
        if not df_e1.empty:
            log.info("\n" + str(
                df_e1.groupby(["Group", "Algorithm"])["Victims_pct"]
                     .agg(["mean", "std", "count"]).round(2)
            ))

    if args.only in ("e2", "both"):
        df_e2 = run_experiment_e2(scenarios, seeds, args)
        plot_boxplot(df_e2, GRAPHS_DIR / "exp_60scen_e2.png",
                     "E2 — Composición 5D0P / 3D2P / 0D5P (8 SAREnv scenarios)")
        log.info("\n" + "=" * 70)
        log.info("  RESUMEN E2 — Victims_pct (media ± std)")
        log.info("=" * 70)
        if not df_e2.empty:
            log.info("\n" + str(
                df_e2.groupby(["Group", "Algorithm"])["Victims_pct"]
                     .agg(["mean", "std", "count"]).round(2)
            ))

    total_min = (time.perf_counter() - t_global) / 60.0
    log.info(f"\n>> TOTAL {total_min:.1f} min ({total_min/60.0:.1f} h)")


if __name__ == "__main__":
    main()
