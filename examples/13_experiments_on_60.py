# examples/13_experiments_on_60.py
"""
Validación de los hallazgos de E6–E8 (docs/21) sobre el dataset SAREnv.

Mientras que examples/10_phase4_experiments.py barre los parámetros sobre
UN solo mapa (Maigmo), este script ejecuta una muestra ESTRATIFICADA de
los 60 escenarios SAREnv (2 por grupo: flat/temperate, flat/dry,
mountainous/temperate, mountainous/dry) y compara la configuración
baseline contra las variantes "mejor" identificadas en Maigmo, para
comprobar si los hallazgos generalizan.

Diseño (OAT - one-at-a-time):
    baseline          : comm=500,  evap=0.01,  eep=0.0
    E6_comm_2000      : comm=2000, evap=0.01,  eep=0.0
    E6_comm_5000      : comm=5000, evap=0.01,  eep=0.0
    E7_evap_0.005     : comm=500,  evap=0.005, eep=0.0
    E7_evap_0.002     : comm=500,  evap=0.002, eep=0.0
    E8_eep_0.5        : comm=500,  evap=0.01,  eep=0.5
    E8_eep_1.5        : comm=500,  evap=0.01,  eep=1.5

Total: 8 escenarios × 2 seeds × 7 configs = 112 simulaciones.

Uso:
    python examples/13_experiments_on_60.py                 # Todo
    python examples/13_experiments_on_60.py --quick         # 4 esc × 1 seed
    python examples/13_experiments_on_60.py --configs baseline E6_comm_2000
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

# Reutilizamos las utilidades del script 12 (carga, budget, grupo).
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

# Muestra estratificada (2 por grupo).
SCENARIO_SAMPLE = [1, 5, 16, 20, 31, 35, 46, 50]

# Definición de configuraciones (OAT sobre baseline).
CONFIGS: dict[str, dict] = {
    "baseline":      dict(comm=500.0,  evap=0.01,  alert_evap=0.005,  eep=0.0),
    "E6_comm_2000":  dict(comm=2000.0, evap=0.01,  alert_evap=0.005,  eep=0.0),
    "E6_comm_5000":  dict(comm=5000.0, evap=0.01,  alert_evap=0.005,  eep=0.0),
    "E7_evap_0005":  dict(comm=500.0,  evap=0.005, alert_evap=0.0025, eep=0.0),
    "E7_evap_0002":  dict(comm=500.0,  evap=0.002, alert_evap=0.001,  eep=0.0),
    "E8_eep_05":     dict(comm=500.0,  evap=0.01,  alert_evap=0.005,  eep=0.5),
    "E8_eep_15":     dict(comm=500.0,  evap=0.01,  alert_evap=0.005,  eep=1.5),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validación de E6–E8 sobre 60 escenarios SAREnv"
    )
    p.add_argument("--scenarios", type=int, nargs="*", default=None,
                   help=f"Escenarios a usar (default: {SCENARIO_SAMPLE})")
    p.add_argument("--seeds", type=int, default=2,
                   help="Número de semillas (default: 2)")
    p.add_argument("--configs", type=str, nargs="*", default=None,
                   help=f"Configs a evaluar (default: todas: {list(CONFIGS)})")
    p.add_argument("--num-drones", type=int, default=5)
    p.add_argument("--num-dogs", type=int, default=0)
    p.add_argument("--max-hops", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=15_000)
    p.add_argument("--num-victims", type=int, default=200)
    p.add_argument("--budget-factor", type=float, default=1.0)
    p.add_argument("--quick", action="store_true",
                   help="Modo rápido: 4 escenarios × 1 seed × 3 configs")
    p.add_argument("--tag", type=str, default="",
                   help="Sufijo para el CSV de salida")
    return p.parse_args()


def run_one(item, config_name: str, cfg: dict, seed: int,
            args: argparse.Namespace) -> dict:
    """Ejecuta una simulación y devuelve la fila de métricas."""
    import random
    random.seed(seed)
    np.random.seed(seed)

    # Víctimas (mismas para todas las configs si seed fija → comparable)
    try:
        victim_gen = LostPersonLocationGenerator(item)
        victim_points = victim_gen.generate_locations(
            args.num_victims, percent_random_samples=0
        )
        victims_gdf = gpd.GeoDataFrame(geometry=victim_points,
                                       crs=item.features.crs)
    except Exception as e:
        log.warning(f"  generador víctimas falló: {e}")
        victims_gdf = gpd.GeoDataFrame(columns=["geometry"],
                                       crs=item.features.crs)

    budget = compute_budget(item, args.budget_factor)

    drone_cfg = DroneConfig(altitude=80.0, fov_deg=45.0)
    dog_cfg = RobotDogConfig(sensor_range=20.0)
    drone_cfg.comm_range = cfg["comm"]
    dog_cfg.comm_range = cfg["comm"]
    drone_cfg.ever_explored_penalty = cfg["eep"]
    dog_cfg.ever_explored_penalty = cfg["eep"]

    swarm_cfg = SwarmConfig(
        num_drones=args.num_drones,
        num_dogs=args.num_dogs,
        budget_per_agent=budget,
        max_steps=args.max_steps,
        max_hops=args.max_hops,
        drone_config=drone_cfg,
        dog_config=dog_cfg,
    )
    swarm_cfg.evaporation_rate = cfg["evap"]
    swarm_cfg.alert_evaporation_rate = cfg["alert_evap"]

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
        "Config": config_name,
        "Comm_range": cfg["comm"],
        "Evaporation_rate": cfg["evap"],
        "Ever_explored_penalty": cfg["eep"],
        "Seed": seed,
        "Victims_pct": round(victim_pct, 2),
        "Area_km2": round(area_km2, 2),
        "Coverage_ratio": round(report["coverage_ratio"], 4),
        "Prob_covered_ratio": round(report["probability_coverage_ratio"], 4),
        "Overlap_ratio": round(report["overlap_ratio"], 4),
        "Efficiency_ratio": round(report["efficiency_ratio"], 4),
        "Agent_prob_gini": round(report.get("agent_probability_gini", 0.0), 4),
        "Mean_prob_swept": round(report.get("mean_probability_swept", 0.0), 4),
        "Total_prob_swept": round(report.get("total_probability_swept", 0.0), 4),
        "Likelihood": round(likelihood, 6),
        "Path_km": round(path_km, 2),
        "Elapsed_s": round(elapsed, 1),
    }


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    if args.quick:
        scenarios = [1, 16, 31, 46]
        seeds = SEED_LIST[:1]
        configs = ["baseline", "E6_comm_2000", "E7_evap_0005"]
    else:
        scenarios = args.scenarios or SCENARIO_SAMPLE
        seeds = SEED_LIST[: args.seeds]
        configs = args.configs or list(CONFIGS)

    total = len(scenarios) * len(seeds) * len(configs)
    log.info("=" * 70)
    log.info(f"  VALIDACIÓN E6–E8 SOBRE SAREnv (muestra)")
    log.info(f"  Escenarios: {scenarios}")
    log.info(f"  Seeds: {seeds}")
    log.info(f"  Configs: {configs}")
    log.info(f"  Total simulaciones: {total}")
    log.info("=" * 70)

    rows: list[dict] = []
    t_global = time.perf_counter()
    n_done = 0

    for sid in scenarios:
        log.info(f"\n─── Escenario {sid} ({get_scenario_group(sid)}) ───")
        item = load_scenario_item(sid)
        if item is None:
            log.warning(f"  saltado")
            continue
        log.info(f"  Grid: {item.heatmap.shape}, radius: {item.radius_km:.1f} km")

        for seed in seeds:
            for cfg_name in configs:
                cfg = CONFIGS[cfg_name]
                n_done += 1
                elapsed_min = (time.perf_counter() - t_global) / 60
                eta_min = elapsed_min / n_done * (total - n_done) if n_done else 0
                log.info(
                    f"  [{n_done}/{total}] sid={sid} seed={seed} cfg={cfg_name} "
                    f"(elapsed {elapsed_min:.1f} min, ETA {eta_min:.1f} min)"
                )
                row = run_one(item, cfg_name, cfg, seed, args)
                row.update({
                    "Scenario": sid,
                    "Group": get_scenario_group(sid),
                    "Environment": item.environment_type,
                    "Climate": item.environment_climate,
                    "Grid": f"{item.heatmap.shape[0]}x{item.heatmap.shape[1]}",
                    "Radius_km": item.radius_km,
                })
                rows.append(row)

                # Guardado parcial cada fila (la run es larga)
                suffix = f"_{args.tag}" if args.tag else ""
                pd.DataFrame(rows).to_csv(
                    RESULTS_DIR / f"exp_60scen_partial{suffix}.csv",
                    index=False,
                )

    total_min = (time.perf_counter() - t_global) / 60
    df = pd.DataFrame(rows)
    suffix = f"_{args.tag}" if args.tag else ""
    out = RESULTS_DIR / f"exp_60scen_combined{suffix}.csv"
    df.to_csv(out, index=False)
    log.info(f"\n{'=' * 70}")
    log.info(f"  RESULTADOS: {out}  ({len(df)} filas, {total_min:.1f} min)")
    log.info(f"{'=' * 70}")

    # Resumen agregado por Config
    print("\n" + "=" * 70)
    print("  RESUMEN GLOBAL POR CONFIG  (media ± std sobre escenarios × seeds)")
    print("=" * 70)
    cols = ["Victims_pct", "Prob_covered_ratio", "Overlap_ratio",
            "Agent_prob_gini", "Efficiency_ratio"]
    agg = df.groupby("Config")[cols].agg(["mean", "std"]).round(3)
    print(agg.to_string())

    # Plot agregado: barplot por Config con error bars
    try:
        plot_summary(df, GRAPHS_DIR / f"exp_60scen_summary{suffix}.png")
    except Exception as e:
        log.warning(f"  plot falló: {e}")


def plot_summary(df: pd.DataFrame, out: Path) -> None:
    metrics = [
        ("Victims_pct", "Víctimas encontradas (%)"),
        ("Prob_covered_ratio", "Prob. cubierta (ratio)"),
        ("Overlap_ratio", "Solapamiento (ratio)"),
        ("Agent_prob_gini", "Gini reparto por agente"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Validación E6–E8 sobre 8 escenarios SAREnv (2 por grupo)",
        fontsize=14, fontweight="bold",
    )
    config_order = list(CONFIGS)
    config_order = [c for c in config_order if c in df["Config"].unique()]

    for ax, (col, label) in zip(axes.flat, metrics):
        agg = df.groupby("Config")[col].agg(["mean", "std"]).reindex(config_order)
        ax.bar(range(len(agg)), agg["mean"], yerr=agg["std"],
               capsize=4, color=["#888"] + ["#2196F3"] * 2 +
                                ["#4CAF50"] * 2 + ["#FF9800"] * 2)
        ax.set_xticks(range(len(agg)))
        ax.set_xticklabels(agg.index, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3, axis="y")
        # Anotaciones
        for i, m in enumerate(agg["mean"]):
            ax.annotate(f"{m:.3f}", (i, m), ha="center", va="bottom",
                        fontsize=8, xytext=(0, 3), textcoords="offset points")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info(f"  >> Gráfica: {out}")


if __name__ == "__main__":
    main()
