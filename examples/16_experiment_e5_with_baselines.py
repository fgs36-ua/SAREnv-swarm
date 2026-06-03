# examples/16_experiment_e5_with_baselines.py
"""
E5' — Resiliencia comparativa: Swarm vs baselines centralizados bajo fallos.

Extiende el E5 original (script 14) añadiendo Pizza y Greedy como baselines
para verificar empíricamente si existe el "punto de cruce" en el que el
enjambre, gracias a su degradación graceful, supera a una ruta precomputada
que pierde agentes sin posibilidad de replanificar.

Modelo de fallo
---------------
- Swarm: a partir de KILL_AT_STEP se llama a `sim.kill_agent(...)` sobre
  `n_kill = round(kill_fraction × n_agents)` agentes (mismo modelo que script 14).
- Pizza/Greedy: planificación centralizada offline. Los baselines NO replanifican
  cuando un agente falla. Modelamos el fallo como **truncamiento por tiempo**
  (opción B, más realista que eliminación completa):
    * Cada baseline genera 5 rutas con el budget E1-equivalente.
    * Las primeras `n_kill` rutas se truncan a `FAIL_PROGRESS × longitud_total`
      donde `FAIL_PROGRESS = KILL_AT_STEP / max_steps ≈ 0.133`.
    * Las rutas no afectadas se mantienen completas.

Diseño:
    8 escenarios estratificados × 3 seeds × 4 kill_fractions × 3 algoritmos
    = 288 simulaciones.
    Algoritmos: Swarm_3D2P, Pizza, Greedy.
    kill_fractions: {0.0, 0.2, 0.4, 0.6} (igual que script 14).

Salida:
    results/exp_60scen_e5_baselines.csv  — métricas por corrida
    graphs/exp_60scen_e5_baselines.png   — APD vs kill_fraction (3 curvas)

Uso:
    python examples/16_experiment_e5_with_baselines.py
    python examples/16_experiment_e5_with_baselines.py --quick   # 4 scen × 1 seed
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
from shapely.ops import substring

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
SCENARIO_SAMPLE = [1, 5, 16, 20, 31, 35, 46, 50]
KILL_CONFIGS = [0.0, 0.2, 0.4, 0.6]
KILL_AT_STEP = 2000  # coherente con script 14


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="E5' — Resiliencia comparativa con baselines"
    )
    p.add_argument("--scenarios", type=int, nargs="*", default=None)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=15_000)
    p.add_argument("--num-victims", type=int, default=200)
    p.add_argument("--budget-factor", type=float, default=1.0)
    p.add_argument("--num-drones", type=int, default=3,
                   help="Drones del swarm (default 3)")
    p.add_argument("--num-dogs", type=int, default=2,
                   help="Perros del swarm (default 2)")
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


# ──────────────────────────────────────────────────────────────────────
#  SWARM bajo fallo (mismo patrón que script 14)
# ──────────────────────────────────────────────────────────────────────
def run_swarm_with_kill(item, sid: int, num_drones: int, num_dogs: int,
                        kill_fraction: float, seed: int, args,
                        label: str) -> dict:
    import random
    random.seed(seed)
    np.random.seed(seed)
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

    n_agents = num_drones + num_dogs
    n_kill = int(round(kill_fraction * n_agents))
    killed = False
    n_steps_alive = 0
    for step_i in range(args.max_steps):
        if not killed and step_i >= KILL_AT_STEP and n_kill > 0:
            active_ids = [a.id for a in sim.agents if a.active]
            for aid in active_ids[:n_kill]:
                sim.kill_agent(aid)
            killed = True
        sim.step()
        n_steps_alive = step_i + 1
        if not any(a.active for a in sim.agents):
            break
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
        "Scenario": sid,
        "Group": get_scenario_group(sid),
        "Environment": item.environment_type,
        "Climate": item.environment_climate,
        "Radius_km": item.radius_km,
        "Budget_m": round(budget, 1),
        "Algorithm": label,
        "n_agents": n_agents,
        "Kill_fraction": kill_fraction,
        "N_agents_killed": n_kill,
        "Steps_completed": n_steps_alive,
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


# ──────────────────────────────────────────────────────────────────────
#  BASELINES bajo fallo (truncamiento por tiempo)
# ──────────────────────────────────────────────────────────────────────
def _truncate_killed_paths(paths: list, n_kill: int, fail_progress: float) -> list:
    """Trunca las primeras n_kill rutas a `fail_progress × longitud_total`.

    Modela un fallo a t = fail_progress × tiempo_total: el agente recorrió
    esa fracción de su ruta antes de caer; las restantes siguen completas.
    """
    if n_kill <= 0 or not paths:
        return paths
    out = []
    for idx, line in enumerate(paths):
        if idx < n_kill and line is not None and not line.is_empty:
            cutoff = max(0.0, line.length * fail_progress)
            if cutoff <= 0:
                # Drone falla antes de moverse: ruta vacía → la omitimos
                continue
            out.append(substring(line, 0, cutoff))
        else:
            out.append(line)
    return out


def run_baseline_with_kill(item, sid: int, algo_name: str, algo_func,
                           n_agents: int, kill_fraction: float,
                           seed: int, args) -> dict:
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

    # Mismo fix de budget que script 15: pasamos budget*n_agents porque los
    # baselines lo dividen internamente por num_drones.
    total_budget = budget * n_agents
    cfg = PathGeneratorConfig(num_drones=n_agents, budget=total_budget)
    gen = PathGenerator(name=algo_name, func=algo_func, path_generator_config=cfg)

    n_kill = int(round(kill_fraction * n_agents))
    fail_progress = KILL_AT_STEP / args.max_steps  # ≈ 0.133

    t0 = time.perf_counter()
    try:
        paths = gen(
            center_x=center_x, center_y=center_y, max_radius=max_radius,
            probability_map=item.heatmap, bounds=item.bounds,
        )
        if isinstance(paths, list):
            paths_eff = _truncate_killed_paths(paths, n_kill, fail_progress)
        else:
            paths_eff = paths
        elapsed = time.perf_counter() - t0

        evaluator = PathEvaluator(
            heatmap=item.heatmap, extent=item.bounds, victims=victims_gdf,
            fov_deg=45.0, altitude=80.0, meters_per_bin=meters_per_bin,
        )
        pe = evaluator.calculate_all_metrics(paths_eff, discount_factor=1.0)
        victim_pct = pe["victim_detection_metrics"].get("percentage_found", 0)
        area_km2 = pe["area_covered"]
        likelihood = pe["total_likelihood_score"]
        path_km = pe["total_path_length"]
    except Exception as e:
        log.warning(f"  Baseline {algo_name} sid={sid} kf={kill_fraction} falló: {e}")
        elapsed = 0
        victim_pct = area_km2 = likelihood = path_km = 0

    return {
        "Scenario": sid,
        "Group": get_scenario_group(sid),
        "Environment": item.environment_type,
        "Climate": item.environment_climate,
        "Radius_km": item.radius_km,
        "Budget_m": round(budget, 1),
        "Algorithm": algo_name,
        "n_agents": n_agents,
        "Kill_fraction": kill_fraction,
        "N_agents_killed": n_kill,
        "Steps_completed": args.max_steps,  # baselines no tienen noción de step
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


# ──────────────────────────────────────────────────────────────────────
#  EXPERIMENTO
# ──────────────────────────────────────────────────────────────────────
def run_experiment(scenarios, seeds, args) -> pd.DataFrame:
    from sarenv.analytics import paths as path_algorithms

    baselines = {
        "Greedy": path_algorithms.generate_greedy_path,
        "Pizza": path_algorithms.generate_pizza_zigzag_path,
    }
    swarm_label = f"Swarm_{args.num_drones}D{args.num_dogs}P"
    n_agents = args.num_drones + args.num_dogs

    rows: list[dict] = []
    total = len(scenarios) * len(seeds) * len(KILL_CONFIGS) * 3
    i = 0
    t_start = time.perf_counter()
    out_partial = RESULTS_DIR / "exp_60scen_e5_baselines_partial.csv"

    log.info("=" * 70)
    log.info(f"E5' — {total} simulaciones planificadas")
    log.info(f"  Algoritmos: {swarm_label}, Pizza, Greedy")
    log.info(f"  kill_fractions: {KILL_CONFIGS}")
    log.info(f"  KILL_AT_STEP={KILL_AT_STEP}, fail_progress≈"
             f"{KILL_AT_STEP/args.max_steps:.3f}")
    log.info("=" * 70)

    for sid in scenarios:
        item = load_scenario_item(sid)
        if item is None:
            log.error(f"E5' sid={sid}: imposible cargar, saltando")
            continue
        log.info(f"\n─── Escenario {sid} ({get_scenario_group(sid)}) ───")
        for seed in seeds:
            for kf in KILL_CONFIGS:
                # Swarm
                i += 1
                el = (time.perf_counter() - t_start) / 60.0
                eta = el / i * (total - i) if i else 0
                log.info(f"  [E5' {i}/{total}] sid={sid} seed={seed} "
                         f"kill={kf:.0%} {swarm_label} "
                         f"(elapsed {el:.1f} min, ETA {eta:.1f} min)")
                rows.append(run_swarm_with_kill(
                    item, sid, args.num_drones, args.num_dogs,
                    kf, seed, args, swarm_label,
                ))
                pd.DataFrame(rows).to_csv(out_partial, index=False)
                # Baselines
                for algo_name, algo_func in baselines.items():
                    i += 1
                    el = (time.perf_counter() - t_start) / 60.0
                    eta = el / i * (total - i) if i else 0
                    log.info(f"  [E5' {i}/{total}] sid={sid} seed={seed} "
                             f"kill={kf:.0%} {algo_name} "
                             f"(elapsed {el:.1f} min, ETA {eta:.1f} min)")
                    rows.append(run_baseline_with_kill(
                        item, sid, algo_name, algo_func,
                        n_agents, kf, seed, args,
                    ))
                    pd.DataFrame(rows).to_csv(out_partial, index=False)

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "exp_60scen_e5_baselines.csv"
    df.to_csv(out_csv, index=False)
    log.info(f"\n  >> CSV: {out_csv} ({len(df)} filas)")
    return df


# ──────────────────────────────────────────────────────────────────────
#  PLOT
# ──────────────────────────────────────────────────────────────────────
def plot_results(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    agg = (df.groupby(["Algorithm", "Kill_fraction"])["Victims_pct"]
             .agg(["mean", "std"]).reset_index())
    colors = {"Pizza": "#d62728", "Greedy": "#2ca02c"}
    for algo in agg["Algorithm"].unique():
        sub = agg[agg["Algorithm"] == algo].sort_values("Kill_fraction")
        color = colors.get(algo, "#1f77b4")
        ax.errorbar(sub["Kill_fraction"] * 100, sub["mean"], yerr=sub["std"],
                    marker="o", capsize=4, linewidth=2, label=algo, color=color)
    ax.set_xlabel("Fracción de agentes eliminados (%)")
    ax.set_ylabel("Víctimas encontradas (%)")
    ax.set_title("E5' — Degradación por fallos: Swarm vs Pizza vs Greedy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    log.info(f"  >> Plot: {out}")


# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    GRAPHS_DIR.mkdir(exist_ok=True)

    scenarios = args.scenarios or SCENARIO_SAMPLE
    seeds = SEED_LIST[:args.seeds]
    if args.quick:
        scenarios = scenarios[:4]
        seeds = seeds[:1]
        log.info("MODO QUICK: 4 escenarios × 1 seed")

    df = run_experiment(scenarios, seeds, args)
    plot_results(df, GRAPHS_DIR / "exp_60scen_e5_baselines.png")

    log.info("\n" + "=" * 70)
    log.info("RESUMEN E5'")
    log.info("=" * 70)
    summary = (df.groupby(["Algorithm", "Kill_fraction"])["Victims_pct"]
                 .agg(["mean", "std", "count"]).round(2))
    log.info("\n" + summary.to_string())


if __name__ == "__main__":
    main()
