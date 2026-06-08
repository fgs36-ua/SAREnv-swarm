# examples/15_experiment10_load_balance_comparative.py
"""
E10 — Reparto de carga comparativo: Swarm vs Pizza vs Greedy vs Spiral.

Extiende E9 (que sólo medía el Gini del enjambre) a los algoritmos
centralizados, calculando por agente (= por path en los baselines)
la probabilidad cubierta de forma única (suma de ``item.heatmap[r, c]``
sobre las celdas observadas únicas) y el índice de Gini sobre el
reparto resultante.

Hipótesis a contrastar (docs/21):
  - Pizza: Gini ≈ 0 (sectores iguales por construcción, mismo trozo de masa).
  - Greedy: Gini alto (>0.3): el primer agente acapara la zona caliente.
  - Swarm: Gini bajo-medio (0.05–0.15): coordinación implícita.

Salidas
-------
results/exp10_load_balance_comparative.csv
    Resumen por (seed, algoritmo): n_agents, mean, total, gini, victims_pct.
results/exp10_load_balance_comparative_per_agent.csv
    Detalle por agente: seed, algorithm, agent_id, prob_swept, cells_observed.
graphs/exp10_load_balance_comparative.{pdf,png}
    Tres paneles: media por agente ± σ; índice de Gini; barras por agente
    para una semilla representativa.

Uso
---
    python examples/15_experiment10_load_balance_comparative.py --seeds 3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import LineString

import sarenv
from sarenv.analytics import paths as path_algorithms
from sarenv.analytics.metrics import PathEvaluator
from sarenv.swarm.comparative import SwarmComparativeEvaluator
from sarenv.swarm.config import SwarmConfig, DroneConfig, RobotDogConfig
from sarenv.swarm.metrics import SwarmMetrics
from sarenv.swarm.simulator import SwarmSimulator
from sarenv.utils.logging_setup import get_logger

log = get_logger()

RESULTS_DIR = Path("results")
GRAPHS_DIR = Path("graphs")
RESULTS_DIR.mkdir(exist_ok=True)
GRAPHS_DIR.mkdir(exist_ok=True)

SEED_LIST = [42, 123, 456, 789, 2025]

BASELINE_ALGORITHMS = {
    "Pizza": path_algorithms.generate_pizza_zigzag_path,
    "Greedy": path_algorithms.generate_greedy_path,
    "Spiral": path_algorithms.generate_spiral_path,
}


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def gini_coefficient(values: list[float]) -> float:
    """Gini sobre lista no negativa. 0 = uniforme, 1 = un único agente acapara."""
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return 0.0
    arr = np.sort(np.clip(arr, 0.0, None))
    n = arr.size
    total = arr.sum()
    if total <= 0:
        return 0.0
    g = (2.0 * float(np.sum(np.arange(1, n + 1) * arr))) / (n * total)
    return float(g - (n + 1.0) / n)


def per_path_prob_swept(
    pe: PathEvaluator, path: LineString,
) -> tuple[float, int]:
    """Probabilidad acumulada (sobre celdas únicas observadas) y nº celdas."""
    if path.is_empty or path.length == 0:
        return 0.0, 0
    num_points = int(np.ceil(path.length / pe.interpolation_resolution)) + 1
    distances = np.linspace(0, path.length, num_points)
    observed: set[tuple[int, int]] = set()
    for d in distances:
        p = path.interpolate(d)
        observed.update(pe.get_visible_cells(p.x, p.y))
    if not observed:
        return 0.0, 0
    prob = float(sum(pe.heatmap[r, c] for (r, c) in observed))
    return prob, len(observed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E10 — Reparto de carga comparativo")
    p.add_argument("--dataset", type=str, default="maigmo_dataset")
    p.add_argument("--size", type=str, default="medium")
    p.add_argument("--budget", type=float, default=100_000,
                   help="Budget por agente en metros (default: 100 km)")
    p.add_argument("--max_steps", type=int, default=15_000)
    p.add_argument("--num_victims", type=int, default=200)
    p.add_argument("--seeds", type=int, default=3,
                   help="Número de semillas")
    p.add_argument("--n_agents", type=int, default=5,
                   help="Agentes por simulación (drones)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────
# Experimento
# ──────────────────────────────────────────────────────────────────

def run_swarm(
    item, victims_gdf, n_drones: int, budget_per_agent: float,
    max_steps: int, seed: int,
) -> tuple[dict[str, float], dict[str, int], dict]:
    """Ejecuta el enjambre y devuelve (prob_swept_por_agente, cells_por_agente,
    summary_metrics_basicos)."""
    drone_cfg = DroneConfig()
    dog_cfg = RobotDogConfig()
    config = SwarmConfig(
        num_drones=n_drones,
        num_dogs=0,
        budget_per_agent=budget_per_agent,
        max_steps=max_steps,
        max_hops=1,
        drone_config=drone_cfg,
        dog_config=dog_cfg,
    )
    sim = SwarmSimulator.from_dataset_item(item, config, seed=seed)
    sim.run()

    # Probabilidad por agente sobre las celdas únicas exploradas, usando el
    # heatmap CRUDO (item.heatmap) — la misma escala que usaremos para los
    # paths centralizados con PathEvaluator.
    raw = item.heatmap
    prob_per_agent: dict[str, float] = {}
    cells_per_agent: dict[str, int] = {}
    for ag in sim.agents:
        cells = ag.cells_ever_explored
        prob_per_agent[ag.id] = float(sum(raw[r, c] for (r, c) in cells))
        cells_per_agent[ag.id] = len(cells)

    sm = SwarmMetrics(sim, victims=victims_gdf)
    report = sm.full_report()
    pe_metrics = sm.evaluate_with_path_evaluator()
    summary = {
        "Victims_pct": pe_metrics.get("victim_detection_metrics", {}).get(
            "percentage_found", 0.0
        ),
        "Coverage_ratio": report["coverage_ratio"],
        "Prob_covered_ratio": report["probability_coverage_ratio"],
    }
    return prob_per_agent, cells_per_agent, summary


def run_centralized(
    item, victims_gdf, algo_name: str, algo_func,
    n_agents: int, budget_per_agent: float, fov_deg: float, altitude: float,
) -> tuple[dict[str, float], dict[str, int], dict]:
    """Ejecuta un algoritmo centralizado y devuelve métricas equivalentes."""
    bounds = item.bounds
    cx = (bounds[0] + bounds[2]) / 2
    cy = (bounds[1] + bounds[3]) / 2
    max_radius = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) / 2
    total_budget = budget_per_agent * n_agents

    paths_result = algo_func(
        center_x=cx, center_y=cy, max_radius=max_radius,
        num_drones=n_agents, fov_deg=fov_deg, altitude=altitude,
        overlap=0.0, path_point_spacing_m=10.0, budget=total_budget,
        probability_map=item.heatmap, bounds=bounds,
        border_gap_m=15.0, transition_distance_m=50.0,
    )

    meters_per_bin = int(np.ceil((bounds[2] - bounds[0]) / item.heatmap.shape[1]))
    pe = PathEvaluator(
        heatmap=item.heatmap, extent=bounds, victims=victims_gdf,
        fov_deg=fov_deg, altitude=altitude, meters_per_bin=meters_per_bin,
    )

    prob_per_agent: dict[str, float] = {}
    cells_per_agent: dict[str, int] = {}
    for i, path in enumerate(paths_result):
        aid = f"{algo_name.lower()}_{i}"
        prob, n_cells = per_path_prob_swept(pe, path)
        prob_per_agent[aid] = prob
        cells_per_agent[aid] = n_cells

    # Métricas globales del algoritmo via PathEvaluator
    pe_metrics = pe.calculate_all_metrics(paths_result, discount_factor=0.999)
    summary = {
        "Victims_pct": pe_metrics["victim_detection_metrics"].get(
            "percentage_found", 0.0
        ),
        "Coverage_ratio": None,        # no aplicable directamente
        "Prob_covered_ratio": None,
    }
    return prob_per_agent, cells_per_agent, summary


def experiment_10(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("=" * 70)
    log.info("  EXPERIMENTO 10: Reparto de carga comparativo Swarm vs Centralizados")
    log.info("=" * 70)

    seeds = SEED_LIST[: args.seeds]
    summary_rows: list[dict] = []
    per_agent_rows: list[dict] = []

    # Para reusar el cargador de escenario y los defaults de FOV/altitud:
    base_eval = SwarmComparativeEvaluator(
        dataset_dir=args.dataset, size=args.size,
        num_victims=args.num_victims, seeds=seeds,
        budget_per_agent=args.budget,
        swarm_configs=[],  # no nos hace falta ejecutar nada desde aquí
    )

    for seed in seeds:
        log.info(f"  ── Seed {seed}")
        item, victims_gdf = base_eval._load_scenario(seed)
        if item is None:
            log.warning(f"    Escenario no cargado; salto seed={seed}")
            continue

        # 1) Swarm
        t0 = time.perf_counter()
        prob_sw, cells_sw, summ_sw = run_swarm(
            item, victims_gdf,
            n_drones=args.n_agents, budget_per_agent=args.budget,
            max_steps=args.max_steps, seed=seed,
        )
        elapsed_sw = time.perf_counter() - t0
        _record(
            summary_rows, per_agent_rows, seed, "Swarm",
            args.n_agents, prob_sw, cells_sw, summ_sw, elapsed_sw,
        )

        # 2) Algoritmos centralizados
        for algo_name, algo_func in BASELINE_ALGORITHMS.items():
            t0 = time.perf_counter()
            prob_c, cells_c, summ_c = run_centralized(
                item, victims_gdf, algo_name, algo_func,
                n_agents=args.n_agents, budget_per_agent=args.budget,
                fov_deg=base_eval.fov_deg, altitude=base_eval.altitude,
            )
            elapsed_c = time.perf_counter() - t0
            _record(
                summary_rows, per_agent_rows, seed, algo_name,
                args.n_agents, prob_c, cells_c, summ_c, elapsed_c,
            )

    df_sum = pd.DataFrame(summary_rows)
    df_pa = pd.DataFrame(per_agent_rows)

    csv_sum = RESULTS_DIR / "exp10_load_balance_comparative.csv"
    csv_pa = RESULTS_DIR / "exp10_load_balance_comparative_per_agent.csv"
    df_sum.to_csv(csv_sum, index=False)
    df_pa.to_csv(csv_pa, index=False)
    log.info(f"  >> Resumen      → {csv_sum}")
    log.info(f"  >> Detalle por agente → {csv_pa}")
    return df_sum, df_pa


def _record(
    summary_rows: list[dict], per_agent_rows: list[dict],
    seed: int, algorithm: str, n_agents: int,
    prob_per_agent: dict[str, float], cells_per_agent: dict[str, int],
    summary: dict, elapsed_s: float,
) -> None:
    sweeps = list(prob_per_agent.values())
    summary_rows.append({
        "Seed": seed,
        "Algorithm": algorithm,
        "N_agents": n_agents,
        "Mean_prob_swept": float(np.mean(sweeps)) if sweeps else 0.0,
        "Total_prob_swept": float(np.sum(sweeps)) if sweeps else 0.0,
        "Std_prob_swept": float(np.std(sweeps)) if sweeps else 0.0,
        "Min_prob_swept": float(np.min(sweeps)) if sweeps else 0.0,
        "Max_prob_swept": float(np.max(sweeps)) if sweeps else 0.0,
        "Agent_prob_gini": gini_coefficient(sweeps),
        "Victims_pct": summary.get("Victims_pct"),
        "Coverage_ratio": summary.get("Coverage_ratio"),
        "Prob_covered_ratio": summary.get("Prob_covered_ratio"),
        "Elapsed_s": round(elapsed_s, 1),
    })
    for aid, prob in prob_per_agent.items():
        per_agent_rows.append({
            "Seed": seed,
            "Algorithm": algorithm,
            "Agent_id": aid,
            "Prob_swept": prob,
            "Cells_observed": cells_per_agent.get(aid, 0),
        })


# ──────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────

def plot_experiment_10(df_sum: pd.DataFrame, df_pa: pd.DataFrame) -> None:
    if df_sum.empty:
        log.warning("  Sin datos: no se generan gráficas E10.")
        return

    algo_order = ["Swarm", "Pizza", "Greedy", "Spiral"]
    colors = {"Swarm": "#2196F3", "Pizza": "#9C27B0",
              "Greedy": "#FF9800", "Spiral": "#4CAF50"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(
        "Experimento 10: Reparto de carga (probabilidad por agente) por algoritmo",
        fontsize=14, fontweight="bold",
    )

    # (a) Mean_prob_swept por algoritmo (media ± std entre semillas)
    ax = axes[0]
    g = df_sum.groupby("Algorithm")["Mean_prob_swept"].agg(["mean", "std"])
    g = g.reindex([a for a in algo_order if a in g.index])
    bars = ax.bar(g.index, g["mean"], yerr=g["std"], capsize=5,
                  color=[colors[a] for a in g.index],
                  alpha=0.85, edgecolor="black")
    ax.set_title("Media de probabilidad cubierta por agente", fontsize=11)
    ax.set_ylabel("Mean prob swept (raw heatmap units)")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, g["mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.0f}", ha="center", va="bottom", fontsize=9)

    # (b) Gini por algoritmo
    ax = axes[1]
    g = df_sum.groupby("Algorithm")["Agent_prob_gini"].agg(["mean", "std"])
    g = g.reindex([a for a in algo_order if a in g.index])
    bars = ax.bar(g.index, g["mean"], yerr=g["std"], capsize=5,
                  color=[colors[a] for a in g.index],
                  alpha=0.85, edgecolor="black")
    ax.set_title("Gini del reparto por agente (0 = uniforme)", fontsize=11)
    ax.set_ylabel("Gini")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(0.5, g["mean"].max() * 1.2))
    for bar, val in zip(bars, g["mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    # (c) Por agente (semilla representativa = la primera)
    ax = axes[2]
    if not df_pa.empty:
        seed0 = df_pa["Seed"].iloc[0]
        sub = df_pa[df_pa["Seed"] == seed0].copy()
        # Ordenar por algoritmo y prob descendente dentro de cada algo
        sub["__algo_rank"] = sub["Algorithm"].map(
            {a: i for i, a in enumerate(algo_order)}
        ).fillna(99).astype(int)
        sub = sub.sort_values(["__algo_rank", "Prob_swept"],
                              ascending=[True, False])

        # Bar chart agrupado: una barra por agente, color por algoritmo
        x = np.arange(len(sub))
        bar_colors = [colors.get(a, "#999") for a in sub["Algorithm"]]
        ax.bar(x, sub["Prob_swept"], color=bar_colors, alpha=0.85,
               edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["Agent_id"], rotation=60, ha="right",
                           fontsize=8)
        ax.set_title(f"Detalle por agente (seed {seed0})", fontsize=11)
        ax.set_ylabel("Prob swept")
        ax.grid(axis="y", alpha=0.3)
        # Leyenda
        from matplotlib.patches import Patch
        handles = [Patch(color=c, label=a) for a, c in colors.items()
                   if a in sub["Algorithm"].unique()]
        ax.legend(handles=handles, loc="upper right", fontsize=8)

    plt.tight_layout()
    out = GRAPHS_DIR / "exp10_load_balance_comparative.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  >> Gráfica guardada en {out}")


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()
    df_sum, df_pa = experiment_10(args)
    elapsed = time.perf_counter() - t0

    if df_sum.empty:
        log.error("No se generaron resultados.")
        return 1

    log.info("=" * 70)
    log.info("  Resumen E10")
    log.info("=" * 70)
    cols = ["Algorithm", "Mean_prob_swept", "Total_prob_swept",
            "Agent_prob_gini", "Victims_pct"]
    summary = df_sum.groupby("Algorithm")[cols[1:]].agg(["mean", "std"])
    log.info(f"\n{summary.round(3)}")
    log.info(f"  Tiempo total: {elapsed/60:.1f} min")

    plot_experiment_10(df_sum, df_pa)
    return 0


if __name__ == "__main__":
    sys.exit(main())
