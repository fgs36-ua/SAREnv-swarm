"""
Validación de E3 (max_hops) y E5 (resiliencia) sobre muestra SAREnv.

Continúa la línea de examples/13_experiments_on_60.py: mientras E6–E8 ya
están validados, faltan los hallazgos de Maigmo §6.5 (max_hops) y §6.7
(resiliencia ante fallos) sobre múltiples mapas/grupos.

Diseño:
    Fase E3 — max_hops ∈ {0, 1, 3, 999}
        4 configs × 8 escenarios × 2 seeds = 64 sims
        Baseline: nuevo default (comm_range=2000, evap=0.01, eep=0.0).
        Estudia el espectro descentralizado→cuasi-centralizado.

    Fase E5 — kill_fraction ∈ {0.0, 0.2, 0.4, 0.6}
        4 niveles × 8 escenarios × 2 seeds = 64 sims
        Baseline: 5 drones, max_hops=1, kill_at_step=2000.
        Estudia la degradación del enjambre ante fallos de agentes.

Total: 128 sims, ≈ 10 h con 5 min/sim.

Uso:
    python examples/14_experiments_e3_e5_on_60.py             # ambas fases
    python examples/14_experiments_e3_e5_on_60.py --phase e3  # sólo E3
    python examples/14_experiments_e3_e5_on_60.py --phase e5  # sólo E5
    python examples/14_experiments_e3_e5_on_60.py --quick     # smoke test
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

# Reutilizamos utilidades del script 12.
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

# Nuevo baseline tras docs/21 §6 (comm_range=2000 ya es default).
HOPS_CONFIGS = [0, 1, 3, 999]
KILL_CONFIGS = [0.0, 0.2, 0.4, 0.6]
KILL_AT_STEP = 2000  # mismo que experiment_5 original


# ─────────────────────────────────────────────────────────────────────
#  ESCENARIO / VÍCTIMAS / BUDGET COMPARTIDOS
# ─────────────────────────────────────────────────────────────────────
def make_victims(item, num_victims: int, seed: int):
    """Genera víctimas deterministas por seed (comparables entre configs)."""
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


def make_swarm_cfg(args: argparse.Namespace, budget: float,
                   max_hops: int) -> SwarmConfig:
    """SwarmConfig sobre el NUEVO baseline (comm_range=2000 por default)."""
    drone_cfg = DroneConfig(altitude=80.0, fov_deg=45.0)
    dog_cfg = RobotDogConfig(sensor_range=20.0)
    # No tocamos comm_range / evap / eep: usamos los nuevos defaults
    # confirmados en docs/21 §6.
    swarm_cfg = SwarmConfig(
        num_drones=args.num_drones,
        num_dogs=args.num_dogs,
        budget_per_agent=budget,
        max_steps=args.max_steps,
        max_hops=max_hops,
        drone_config=drone_cfg,
        dog_config=dog_cfg,
    )
    return swarm_cfg


def collect_metrics(sim: SwarmSimulator, victims_gdf, elapsed: float,
                    extra: dict) -> dict:
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
    row = {
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
    row.update(extra)
    return row


# ─────────────────────────────────────────────────────────────────────
#  FASE E3 — max_hops
# ─────────────────────────────────────────────────────────────────────
def run_e3_one(item, max_hops: int, seed: int,
               args: argparse.Namespace) -> dict:
    import random
    random.seed(seed)
    np.random.seed(seed)
    victims = make_victims(item, args.num_victims, seed)
    budget = compute_budget(item, args.budget_factor)
    cfg = make_swarm_cfg(args, budget, max_hops=max_hops)
    t0 = time.perf_counter()
    sim = SwarmSimulator.from_dataset_item(item, cfg, seed=seed)
    sim.run()
    elapsed = time.perf_counter() - t0
    return collect_metrics(sim, victims, elapsed, {
        "Phase": "E3",
        "Config": f"hops_{max_hops if max_hops < 999 else 'inf'}",
        "Max_hops": max_hops,
        "Kill_fraction": 0.0,
        "Seed": seed,
    })


def run_phase_e3(scenarios: list[int], seeds: list[int],
                 args: argparse.Namespace) -> pd.DataFrame:
    total = len(scenarios) * len(seeds) * len(HOPS_CONFIGS)
    log.info("=" * 70)
    log.info(f"  FASE E3 — max_hops sobre {len(scenarios)} escenarios "
             f"× {len(seeds)} seeds × {len(HOPS_CONFIGS)} configs "
             f"= {total} sims")
    log.info("=" * 70)
    rows: list[dict] = []
    t0 = time.perf_counter()
    n = 0
    for sid in scenarios:
        log.info(f"\n─── Escenario {sid} ({get_scenario_group(sid)}) ───")
        item = load_scenario_item(sid)
        if item is None:
            log.warning("  saltado")
            continue
        log.info(f"  Grid: {item.heatmap.shape}, radius: {item.radius_km:.1f} km")
        for seed in seeds:
            for hops in HOPS_CONFIGS:
                n += 1
                el = (time.perf_counter() - t0) / 60
                eta = el / n * (total - n) if n else 0
                log.info(
                    f"  [E3 {n}/{total}] sid={sid} seed={seed} hops={hops} "
                    f"(elapsed {el:.1f} min, ETA {eta:.1f} min)"
                )
                row = run_e3_one(item, hops, seed, args)
                row.update({
                    "Scenario": sid,
                    "Group": get_scenario_group(sid),
                    "Environment": item.environment_type,
                    "Climate": item.environment_climate,
                    "Grid": f"{item.heatmap.shape[0]}x{item.heatmap.shape[1]}",
                    "Radius_km": item.radius_km,
                })
                rows.append(row)
                pd.DataFrame(rows).to_csv(
                    RESULTS_DIR / "exp_60scen_e3_partial.csv", index=False
                )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "exp_60scen_e3.csv", index=False)
    log.info(f"  >> E3 CSV: results/exp_60scen_e3.csv ({len(df)} filas)")
    return df


# ─────────────────────────────────────────────────────────────────────
#  FASE E5 — resiliencia (kill_fraction)
# ─────────────────────────────────────────────────────────────────────
def run_e5_one(item, kill_fraction: float, seed: int,
               args: argparse.Namespace) -> dict:
    import random
    random.seed(seed)
    np.random.seed(seed)
    victims = make_victims(item, args.num_victims, seed)
    budget = compute_budget(item, args.budget_factor)
    cfg = make_swarm_cfg(args, budget, max_hops=args.max_hops)

    t0 = time.perf_counter()
    sim = SwarmSimulator.from_dataset_item(item, cfg, seed=seed)

    n_agents = args.num_drones + args.num_dogs
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

    return collect_metrics(sim, victims, elapsed, {
        "Phase": "E5",
        "Config": f"kill_{int(kill_fraction * 100):02d}",
        "Max_hops": args.max_hops,
        "Kill_fraction": kill_fraction,
        "Kill_at_step": KILL_AT_STEP,
        "N_agents_killed": n_kill,
        "Steps_completed": n_steps_alive,
        "Seed": seed,
    })


def run_phase_e5(scenarios: list[int], seeds: list[int],
                 args: argparse.Namespace) -> pd.DataFrame:
    total = len(scenarios) * len(seeds) * len(KILL_CONFIGS)
    log.info("=" * 70)
    log.info(f"  FASE E5 — kill_fraction sobre {len(scenarios)} escenarios "
             f"× {len(seeds)} seeds × {len(KILL_CONFIGS)} niveles "
             f"= {total} sims")
    log.info("=" * 70)
    rows: list[dict] = []
    t0 = time.perf_counter()
    n = 0
    for sid in scenarios:
        log.info(f"\n─── Escenario {sid} ({get_scenario_group(sid)}) ───")
        item = load_scenario_item(sid)
        if item is None:
            log.warning("  saltado")
            continue
        log.info(f"  Grid: {item.heatmap.shape}, radius: {item.radius_km:.1f} km")
        for seed in seeds:
            for kf in KILL_CONFIGS:
                n += 1
                el = (time.perf_counter() - t0) / 60
                eta = el / n * (total - n) if n else 0
                log.info(
                    f"  [E5 {n}/{total}] sid={sid} seed={seed} kill={kf:.0%} "
                    f"(elapsed {el:.1f} min, ETA {eta:.1f} min)"
                )
                row = run_e5_one(item, kf, seed, args)
                row.update({
                    "Scenario": sid,
                    "Group": get_scenario_group(sid),
                    "Environment": item.environment_type,
                    "Climate": item.environment_climate,
                    "Grid": f"{item.heatmap.shape[0]}x{item.heatmap.shape[1]}",
                    "Radius_km": item.radius_km,
                })
                rows.append(row)
                pd.DataFrame(rows).to_csv(
                    RESULTS_DIR / "exp_60scen_e5_partial.csv", index=False
                )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "exp_60scen_e5.csv", index=False)
    log.info(f"  >> E5 CSV: results/exp_60scen_e5.csv ({len(df)} filas)")
    return df


# ─────────────────────────────────────────────────────────────────────
#  PLOTS
# ─────────────────────────────────────────────────────────────────────
def plot_e3(df: pd.DataFrame, out: Path) -> None:
    metrics = [
        ("Victims_pct", "Víctimas encontradas (%)"),
        ("Prob_covered_ratio", "Prob. cubierta (ratio)"),
        ("Overlap_ratio", "Solapamiento (ratio)"),
        ("Agent_prob_gini", "Gini reparto por agente"),
    ]
    config_order = [f"hops_{h if h < 999 else 'inf'}" for h in HOPS_CONFIGS]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("E3 max_hops sobre 8 escenarios SAREnv (2 por grupo)",
                 fontsize=14, fontweight="bold")
    for ax, (col, label) in zip(axes.flat, metrics):
        agg = df.groupby("Config")[col].agg(["mean", "std"]).reindex(config_order)
        ax.bar(range(len(agg)), agg["mean"], yerr=agg["std"], capsize=4,
               color=["#888", "#2196F3", "#4CAF50", "#FF9800"])
        ax.set_xticks(range(len(agg)))
        ax.set_xticklabels(agg.index, rotation=0, fontsize=10)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3, axis="y")
        for i, m in enumerate(agg["mean"]):
            ax.annotate(f"{m:.3f}", (i, m), ha="center", va="bottom",
                        fontsize=8, xytext=(0, 3), textcoords="offset points")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info(f"  >> Gráfica E3: {out}")


def plot_e5(df: pd.DataFrame, out: Path) -> None:
    metrics = [
        ("Victims_pct", "Víctimas encontradas (%)"),
        ("Prob_covered_ratio", "Prob. cubierta (ratio)"),
        ("Area_km2", "Área cubierta (km²)"),
        ("Efficiency_ratio", "Eficiencia"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("E5 resiliencia (kill_fraction) sobre 8 escenarios SAREnv",
                 fontsize=14, fontweight="bold")
    # Una línea por grupo para mostrar cómo varía la degradación.
    groups = sorted(df["Group"].dropna().unique())
    colors = {"flat_temperate": "#2196F3", "mountain_temperate": "#9C27B0",
              "flat_dry": "#FF9800", "mountain_dry": "#4CAF50"}
    for ax, (col, label) in zip(axes.flat, metrics):
        for g in groups:
            sub = df[df["Group"] == g]
            agg = sub.groupby("Kill_fraction")[col].agg(["mean", "std"])
            ax.errorbar(agg.index * 100, agg["mean"], yerr=agg["std"],
                        marker="o", linewidth=2, capsize=4,
                        color=colors.get(g, "#444"), label=g)
        # Línea global
        agg_all = df.groupby("Kill_fraction")[col].agg("mean")
        ax.plot(agg_all.index * 100, agg_all.values, "k--", lw=1.5,
                alpha=0.6, label="global")
        ax.set_xlabel("Agentes perdidos (%)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info(f"  >> Gráfica E5: {out}")


# ─────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validación E3/E5 sobre SAREnv")
    p.add_argument("--phase", choices=["e3", "e5", "both"], default="both")
    p.add_argument("--scenarios", type=int, nargs="*", default=None)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--num-drones", type=int, default=5)
    p.add_argument("--num-dogs", type=int, default=0)
    p.add_argument("--max-hops", type=int, default=1,
                   help="max_hops para E5 (E3 lo sobrescribe)")
    p.add_argument("--max-steps", type=int, default=15_000)
    p.add_argument("--num-victims", type=int, default=200)
    p.add_argument("--budget-factor", type=float, default=1.0)
    p.add_argument("--quick", action="store_true",
                   help="Smoke test: 2 escenarios × 1 seed, configs reducidas")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    if args.quick:
        scenarios = [1, 16]
        seeds = SEED_LIST[:1]
        global HOPS_CONFIGS, KILL_CONFIGS  # noqa: PLW0603
        HOPS_CONFIGS = [0, 999]
        KILL_CONFIGS = [0.0, 0.4]
    else:
        scenarios = args.scenarios or SCENARIO_SAMPLE
        seeds = SEED_LIST[: args.seeds]

    log.info("=" * 70)
    log.info(f"  VALIDACIÓN E3 + E5 SOBRE SAREnv")
    log.info(f"  Phase: {args.phase}")
    log.info(f"  Escenarios: {scenarios}")
    log.info(f"  Seeds: {seeds}")
    log.info(f"  Defaults: comm_range=2000 (nuevo baseline docs/21 §6)")
    log.info("=" * 70)

    t_global = time.perf_counter()
    if args.phase in ("e3", "both"):
        df_e3 = run_phase_e3(scenarios, seeds, args)
        try:
            plot_e3(df_e3, GRAPHS_DIR / "exp_60scen_e3.png")
        except Exception as e:
            log.warning(f"  plot E3 falló: {e}")
        print("\n" + "=" * 70)
        print("  RESUMEN E3 — max_hops")
        print("=" * 70)
        cols = ["Victims_pct", "Prob_covered_ratio", "Overlap_ratio",
                "Agent_prob_gini"]
        print(df_e3.groupby("Config")[cols].agg(["mean", "std"]).round(3)
              .to_string())

    if args.phase in ("e5", "both"):
        df_e5 = run_phase_e5(scenarios, seeds, args)
        try:
            plot_e5(df_e5, GRAPHS_DIR / "exp_60scen_e5.png")
        except Exception as e:
            log.warning(f"  plot E5 falló: {e}")
        print("\n" + "=" * 70)
        print("  RESUMEN E5 — kill_fraction")
        print("=" * 70)
        cols = ["Victims_pct", "Prob_covered_ratio", "Area_km2",
                "Efficiency_ratio"]
        print(df_e5.groupby("Kill_fraction")[cols].agg(["mean", "std"])
              .round(3).to_string())

    total_min = (time.perf_counter() - t_global) / 60
    log.info(f"\n>> TOTAL {total_min:.1f} min ({total_min/60:.1f} h)")


if __name__ == "__main__":
    main()
