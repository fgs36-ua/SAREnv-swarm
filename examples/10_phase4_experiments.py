# examples/10_phase4_experiments.py
"""
Fase 4 — Experimentos comparativos del TFG.

Nueve experimentos cubriendo la evaluación completa del enjambre:
  1. Enjambre heterogéneo vs algoritmos centralizados (Greedy, Spiral, Pizza)
  2. Solo drones vs solo perros vs mixto
  3. Impacto de max_hops {0, 1, 3, 999}
  4. Escalabilidad con N agentes (budget total fijo)
  5. Resiliencia ante fallos (kill_fraction {0, 0.2, 0.4, 0.6})
  6. Rango de gossip ``comm_range`` (E6, docs/20)
  7. Persistencia de feromonas (``evaporation_rate``, E7, docs/20)
  8. Hard-mask sobre celdas observadas (``ever_explored_penalty``, E8)
  9. Reparto de carga por agente (índice de Gini, E9, docs/20)

Para cada experimento se repiten varias semillas (--seeds) y se guardan
los resultados en results/ como CSV + gráficas en graphs/.

Uso:
    python examples/10_phase4_experiments.py                   # Todos
    python examples/10_phase4_experiments.py --exp 1           # Solo E1
    python examples/10_phase4_experiments.py --exp 6 7 8 9     # Solo nuevos
    python examples/10_phase4_experiments.py --exp 2 --seeds 3 # Exp 2 con 3 semillas
    python examples/10_phase4_experiments.py --budget 200000   # Budget más alto
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

import sarenv
from sarenv.swarm.comparative import SwarmComparativeEvaluator
from sarenv.utils.logging_setup import get_logger

log = get_logger()

RESULTS_DIR = Path("results")
GRAPHS_DIR = Path("graphs")
SEED_LIST = [42, 123, 456, 789, 2025]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fase 4 — Experimentos comparativos")
    p.add_argument("--exp", type=int, nargs="*", default=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                   help="Experimentos a ejecutar (1–9). E6–E9 son las pruebas"
                        " adicionales descritas en docs/20.")
    p.add_argument("--dataset", type=str, default="maigmo_dataset")
    p.add_argument("--size", type=str, default="medium")
    p.add_argument("--budget", type=float, default=100_000,
                   help="Budget por agente en metros (default: 100km)")
    p.add_argument("--max_steps", type=int, default=15_000)
    p.add_argument("--num_victims", type=int, default=200)
    p.add_argument("--seeds", type=int, default=3,
                   help="Número de semillas aleatorias a usar")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
#  EXPERIMENTO 1: Enjambre vs Algoritmos Centralizados
# ═══════════════════════════════════════════════════════════════════

def experiment_1(args: argparse.Namespace) -> pd.DataFrame:
    """Enjambre heterogéneo (5D+0P) vs Greedy, Spiral, Pizza.

    Misma cantidad de agentes, mismo budget, mismas víctimas.
    Demuestra que el enjambre descentralizado compite con los planificadores.
    """
    log.info("=" * 70)
    log.info("  EXPERIMENTO 1: Enjambre vs Algoritmos Centralizados")
    log.info("=" * 70)

    seeds = SEED_LIST[:args.seeds]
    evaluator = SwarmComparativeEvaluator(
        dataset_dir=args.dataset,
        size=args.size,
        num_victims=args.num_victims,
        seeds=seeds,
        budget_per_agent=args.budget,
        swarm_configs=[
            {
                "num_drones": 5, "num_dogs": 0, "max_hops": 1,
                "max_steps": args.max_steps, "label": "Swarm_5D",
            },
        ],
    )
    df = evaluator.run_all()

    csv_path = RESULTS_DIR / "exp1_swarm_vs_centralized.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"  >> Resultados guardados en {csv_path}")
    return df


# ═══════════════════════════════════════════════════════════════════
#  EXPERIMENTO 2: Solo drones vs Solo perros vs Mixto
# ═══════════════════════════════════════════════════════════════════

def experiment_2(args: argparse.Namespace) -> pd.DataFrame:
    """Composición del equipo: 5D+0P vs 0D+5P vs 3D+2P.

    Demuestra la complementariedad drone-perro en entorno mixto.
    """
    log.info("=" * 70)
    log.info("  EXPERIMENTO 2: Solo drones vs Solo perros vs Mixto")
    log.info("=" * 70)

    seeds = SEED_LIST[:args.seeds]
    evaluator = SwarmComparativeEvaluator(
        dataset_dir=args.dataset,
        size=args.size,
        num_victims=args.num_victims,
        seeds=seeds,
        budget_per_agent=args.budget,
        swarm_configs=[
            {
                "num_drones": 5, "num_dogs": 0, "max_hops": 1,
                "max_steps": args.max_steps, "label": "Drones_5D",
            },
            {
                "num_drones": 0, "num_dogs": 5, "max_hops": 1,
                "max_steps": args.max_steps, "label": "Dogs_5P",
            },
            {
                "num_drones": 3, "num_dogs": 2, "max_hops": 1,
                "max_steps": args.max_steps, "label": "Mixed_3D2P",
            },
        ],
    )
    df = evaluator.run_all()

    # No incluir los baselines centralizados para este experimento;
    # filtramos solo las filas del enjambre
    swarm_labels = {"Drones_5D", "Dogs_5P", "Mixed_3D2P"}
    df_swarm = df[df["Algorithm"].isin(swarm_labels)].copy()

    csv_path = RESULTS_DIR / "exp2_team_composition.csv"
    df_swarm.to_csv(csv_path, index=False)
    log.info(f"  >> Resultados guardados en {csv_path}")
    return df_swarm


# ═══════════════════════════════════════════════════════════════════
#  EXPERIMENTO 3: Impacto de max_hops (espectro descentralizado)
# ═══════════════════════════════════════════════════════════════════

def experiment_3(args: argparse.Namespace) -> pd.DataFrame:
    """max_hops ∈ {0, 1, 3, 999} con 5 drones.

    max_hops=0 → sin comunicación (agentes independientes)
    max_hops=1 → solo vecinos directos
    max_hops=3 → enjambre realista
    max_hops=999 → cuasi-centralizado (info llega a todos eventualmente)
    """
    log.info("=" * 70)
    log.info("  EXPERIMENTO 3: Impacto de max_hops")
    log.info("=" * 70)

    seeds = SEED_LIST[:args.seeds]
    evaluator = SwarmComparativeEvaluator(
        dataset_dir=args.dataset,
        size=args.size,
        num_victims=args.num_victims,
        seeds=seeds,
        budget_per_agent=args.budget,
        swarm_configs=[
            {
                "num_drones": 5, "num_dogs": 0, "max_hops": 0,
                "max_steps": args.max_steps, "label": "hops_0",
            },
            {
                "num_drones": 5, "num_dogs": 0, "max_hops": 1,
                "max_steps": args.max_steps, "label": "hops_1",
            },
            {
                "num_drones": 5, "num_dogs": 0, "max_hops": 3,
                "max_steps": args.max_steps, "label": "hops_3",
            },
            {
                "num_drones": 5, "num_dogs": 0, "max_hops": 999,
                "max_steps": args.max_steps, "label": "hops_inf",
            },
        ],
    )
    df = evaluator.run_all()

    # Filtrar solo configuraciones de enjambre
    hop_labels = {"hops_0", "hops_1", "hops_3", "hops_inf"}
    df_hops = df[df["Algorithm"].isin(hop_labels)].copy()

    csv_path = RESULTS_DIR / "exp3_max_hops.csv"
    df_hops.to_csv(csv_path, index=False)
    log.info(f"  >> Resultados guardados en {csv_path}")
    return df_hops


# ═══════════════════════════════════════════════════════════════════
#  GENERACIÓN DE GRÁFICAS
# ═══════════════════════════════════════════════════════════════════

def plot_experiment_1(df: pd.DataFrame) -> None:
    """Gráfica de barras agrupadas: Swarm vs Greedy vs Spiral vs Pizza."""
    metrics = [
        ("Likelihood", "Likelihood Score L(π)", "Probabilidad acumulada"),
        ("Victims_pct", "Victims Found D(π) (%)", "Víctimas encontradas (%)"),
        ("Area_km2", "Area Covered (km²)", "Área cubierta (km²)"),
        ("Path_length_km", "Total Path Length (km)", "Longitud total (km)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Experimento 1: Enjambre vs Algoritmos Centralizados",
        fontsize=15, fontweight="bold",
    )

    for ax, (col, title, ylabel) in zip(axes.flat, metrics):
        grouped = df.groupby("Algorithm")[col].agg(["mean", "std"]).reindex(
            ["Swarm_5D", "Greedy", "Spiral", "Pizza"]
        )
        colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]
        bars = ax.bar(
            grouped.index, grouped["mean"],
            yerr=grouped["std"], capsize=5,
            color=colors[:len(grouped)], alpha=0.85, edgecolor="black",
        )
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
        # Valores sobre las barras
        for bar, val in zip(bars, grouped["mean"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = GRAPHS_DIR / "exp1_swarm_vs_centralized.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  >> Gráfica guardada en {out}")


def plot_experiment_2(df: pd.DataFrame) -> None:
    """Gráfica comparativa por composición de equipo."""
    metrics = [
        ("Coverage_ratio", "Coverage Ratio", "Cobertura (ratio)"),
        ("Prob_covered_ratio", "Probability Covered", "Prob. cubierta (ratio)"),
        ("Victims_pct", "Victims Found (%)", "Víctimas encontradas (%)"),
        ("Overlap_ratio", "Overlap Ratio", "Solapamiento (ratio)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Experimento 2: Composición del Equipo",
        fontsize=15, fontweight="bold",
    )

    for ax, (col, title, ylabel) in zip(axes.flat, metrics):
        grouped = df.groupby("Algorithm")[col].agg(["mean", "std"]).reindex(
            ["Drones_5D", "Dogs_5P", "Mixed_3D2P"]
        )
        colors = ["#2196F3", "#8BC34A", "#FF5722"]
        bars = ax.bar(
            grouped.index, grouped["mean"],
            yerr=grouped["std"], capsize=5,
            color=colors[:len(grouped)], alpha=0.85, edgecolor="black",
        )
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, grouped["mean"]):
            if pd.notna(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = GRAPHS_DIR / "exp2_team_composition.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  >> Gráfica guardada en {out}")


def plot_experiment_3(df: pd.DataFrame) -> None:
    """Gráfica de evolución con max_hops en eje X."""
    metrics = [
        ("Coverage_ratio", "Coverage Ratio"),
        ("Prob_covered_ratio", "Prob. Covered"),
        ("Overlap_ratio", "Overlap Ratio"),
        ("Victims_pct", "Victims Found (%)"),
    ]

    hop_order = ["hops_0", "hops_1", "hops_3", "hops_inf"]
    hop_labels = ["0", "1", "3", "∞"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Experimento 3: Impacto de max_hops en el Enjambre",
        fontsize=15, fontweight="bold",
    )

    for ax, (col, title) in zip(axes.flat, metrics):
        means, stds = [], []
        for hop in hop_order:
            vals = df[df["Algorithm"] == hop][col].dropna()
            means.append(vals.mean() if len(vals) > 0 else 0)
            stds.append(vals.std() if len(vals) > 1 else 0)

        ax.errorbar(
            hop_labels, means, yerr=stds,
            marker="o", linewidth=2, capsize=5,
            color="#2196F3", markersize=8,
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("max_hops")
        ax.grid(alpha=0.3)
        for i, (x, m) in enumerate(zip(hop_labels, means)):
            ax.annotate(f"{m:.3f}", (x, m), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9)

    plt.tight_layout()
    out = GRAPHS_DIR / "exp3_max_hops.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  >> Gráfica guardada en {out}")


# ═══════════════════════════════════════════════════════════════════
#  EXPERIMENTO 4: Escalabilidad (N variable)
# ═══════════════════════════════════════════════════════════════════

def experiment_4(args: argparse.Namespace) -> pd.DataFrame:
    """Escalabilidad con BUDGET TOTAL FIJO en mapa grande.

    Presupuesto total = 500 km repartido entre N agentes.
    El swarm reparte N agentes con budget/N cada uno; los centralizados
    reciben el mismo total.  Muestra quién aprovecha mejor un presupuesto
    fijo al añadir agentes.

    Mapa 'large' (13×13 km, 169 km²) → ningún algoritmo satura.
    """
    EXP4_SIZE = "large"
    TOTAL_BUDGET = 500_000  # 500 km fijos
    log.info("=" * 70)
    log.info(f"  EXPERIMENTO 4: Escalabilidad (budget total={TOTAL_BUDGET/1000:.0f}km, size={EXP4_SIZE})")
    log.info("=" * 70)

    seeds = SEED_LIST[:args.seeds]
    agent_counts = [3, 5, 7, 10, 15]
    rows: list[dict] = []

    for n in agent_counts:
        budget_per = TOTAL_BUDGET / n
        log.info(f"  --- N = {n}  (budget/agente = {budget_per/1000:.0f} km) ---")
        n_drones = max(1, round(n * 0.6))
        n_dogs = n - n_drones

        evaluator = SwarmComparativeEvaluator(
            dataset_dir=args.dataset,
            size=EXP4_SIZE,
            num_victims=args.num_victims,
            seeds=seeds,
            budget_per_agent=budget_per,
            swarm_configs=[
                {
                    "num_drones": n_drones, "num_dogs": n_dogs,
                    "max_hops": 1, "max_steps": args.max_steps,
                    "label": f"Swarm_{n}",
                },
            ],
        )
        df_n = evaluator.run_all()
        df_n["N"] = n
        df_n["Budget_per_agent_km"] = budget_per / 1000
        rows.append(df_n)

    df = pd.concat(rows, ignore_index=True)
    csv_path = RESULTS_DIR / "exp4_scalability.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"  >> Resultados guardados en {csv_path}")
    return df


def plot_experiment_4(df: pd.DataFrame) -> None:
    """Gráfica de escalabilidad: swarm vs baselines por N agentes."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Experimento 4: Escalabilidad — Budget total fijo (500 km), mapa 13×13 km",
        fontsize=14, fontweight="bold",
    )

    metrics = [
        ("Victims_pct", "Víctimas encontradas (%)"),
        ("Area_km2", "Área cubierta (km²)"),
        ("Likelihood", "Likelihood L(π)"),
    ]

    for ax, (col, ylabel) in zip(axes.flat, metrics):
        # Swarm: una línea con diferentes N
        swarm_df = df[df["Algorithm"].str.startswith("Swarm")]
        swarm_agg = swarm_df.groupby("N")[col].agg(["mean", "std"])

        ax.errorbar(
            swarm_agg.index, swarm_agg["mean"], yerr=swarm_agg["std"],
            marker="o", linewidth=2.5, capsize=5, color="#2196F3",
            markersize=8, label="Swarm (mixto)", zorder=5,
        )

        # Baselines: líneas horizontales o con pendiente para cada N
        for algo, color, ls in [
            ("Pizza", "#9C27B0", "--"),
            ("Spiral", "#4CAF50", "-."),
            ("Greedy", "#FF9800", ":"),
        ]:
            bdf = df[df["Algorithm"] == algo]
            if bdf.empty:
                continue
            bagg = bdf.groupby("N")[col].agg(["mean", "std"])
            ax.errorbar(
                bagg.index, bagg["mean"], yerr=bagg["std"],
                marker="s", linewidth=2, capsize=4, color=color,
                linestyle=ls, markersize=6, label=algo, alpha=0.8,
            )

        ax.set_xlabel("Número de agentes (N)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = GRAPHS_DIR / "exp4_scalability.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  >> Gráfica guardada en {out}")


# ═══════════════════════════════════════════════════════════════════
#  EXPERIMENTO 5: Resiliencia ante fallos
# ═══════════════════════════════════════════════════════════════════

def experiment_5(args: argparse.Namespace) -> pd.DataFrame:
    """Resiliencia: matar 0%, 20%, 40%, 60% de agentes a mitad de misión.

    El swarm re-distribuye implícitamente (los supervivientes siguen
    explorando zonas sin cubrir). Los centralizados pierden los sectores
    asignados a los drones muertos sin posibilidad de re-planificación.
    """
    log.info("=" * 70)
    log.info("  EXPERIMENTO 5: Resiliencia ante fallos de agentes")
    log.info("=" * 70)

    seeds = SEED_LIST[:args.seeds]
    kill_fractions = [0.0, 0.2, 0.4, 0.6]
    rows: list[dict] = []
    n_agents = 5

    swarm_cfg = {
        "num_drones": 3, "num_dogs": 2, "max_hops": 1,
        "max_steps": args.max_steps,
    }

    for kf in kill_fractions:
        log.info(f"  --- kill_fraction = {kf:.0%} ---")
        for seed in seeds:
            item, victims_gdf = SwarmComparativeEvaluator(
                dataset_dir=args.dataset, size=args.size,
                num_victims=args.num_victims, seeds=[seed],
                budget_per_agent=args.budget,
            )._load_scenario(seed)
            if item is None:
                continue

            evaluator = SwarmComparativeEvaluator(
                dataset_dir=args.dataset, size=args.size,
                num_victims=args.num_victims, seeds=[seed],
                budget_per_agent=args.budget,
            )

            # Swarm con fallos
            row_s = evaluator._evaluate_swarm_with_failures(
                item, victims_gdf, swarm_cfg, seed,
                kill_fraction=kf, kill_at_step=2000,
            )
            row_s["Algorithm"] = "Swarm_3D2P"
            row_s["Seed"] = seed
            row_s["kill_fraction"] = kf
            rows.append(row_s)

            # Pizza con fallos (mejor baseline)
            from sarenv.analytics import paths as path_algorithms
            row_b = evaluator._evaluate_baseline_with_failures(
                item, victims_gdf, "Pizza", path_algorithms.generate_pizza_zigzag_path,
                n_agents, seed, kill_fraction=kf, path_fraction_before_kill=0.3,
            )
            row_b["Algorithm"] = "Pizza"
            row_b["Seed"] = seed
            row_b["kill_fraction"] = kf
            rows.append(row_b)

            # Greedy con fallos
            row_g = evaluator._evaluate_baseline_with_failures(
                item, victims_gdf, "Greedy", path_algorithms.generate_greedy_path,
                n_agents, seed, kill_fraction=kf, path_fraction_before_kill=0.3,
            )
            row_g["Algorithm"] = "Greedy"
            row_g["Seed"] = seed
            row_g["kill_fraction"] = kf
            rows.append(row_g)

    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "exp5_resilience.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"  >> Resultados guardados en {csv_path}")
    return df


def plot_experiment_5(df: pd.DataFrame) -> None:
    """Gráfica de degradación por fracción de agentes perdidos."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Experimento 5: Resiliencia — Degradación ante fallos de agentes",
        fontsize=14, fontweight="bold",
    )

    metrics = [
        ("Victims_pct", "Víctimas encontradas (%)"),
        ("Area_km2", "Área cubierta (km²)"),
        ("Likelihood", "Likelihood L(π)"),
    ]

    algo_styles = {
        "Swarm_3D2P": ("#2196F3", "o", "-", "Swarm (3D+2P)"),
        "Pizza": ("#9C27B0", "s", "--", "Pizza"),
        "Greedy": ("#FF9800", "^", ":", "Greedy"),
    }

    for ax, (col, ylabel) in zip(axes.flat, metrics):
        for algo, (color, marker, ls, label) in algo_styles.items():
            adf = df[df["Algorithm"] == algo]
            if adf.empty:
                continue
            agg = adf.groupby("kill_fraction")[col].agg(["mean", "std"])
            x_labels = [f"{kf:.0%}" for kf in agg.index]
            ax.errorbar(
                x_labels, agg["mean"], yerr=agg["std"],
                marker=marker, linewidth=2.5, capsize=5, color=color,
                linestyle=ls, markersize=8, label=label,
            )

        ax.set_xlabel("Fracción de agentes perdidos")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = GRAPHS_DIR / "exp5_resilience.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  >> Gráfica guardada en {out}")


# ═══════════════════════════════════════════════════════════════════
#  EXPERIMENTO 6: Rango de gossip (comm_range)  — docs/20 §E6
# ═══════════════════════════════════════════════════════════════════

def _sweep_swarm(
    args: argparse.Namespace,
    *,
    label_fn,
    overrides_list: list[dict],
    base_cfg: dict | None = None,
    extra_eval_kwargs_fn=lambda o: {},
) -> pd.DataFrame:
    """Helper común para barridos: ejecuta un único swarm_config por valor
    del barrido, sin baselines centralizados (ya están en E1).

    ``overrides_list`` es una lista de dicts; por cada entrada se crea un
    ``SwarmComparativeEvaluator`` con los kwargs devueltos por
    ``extra_eval_kwargs_fn(override)`` y se ejecuta el enjambre. Las filas
    se etiquetan con ``label_fn(override)``.
    """
    seeds = SEED_LIST[:args.seeds]
    base_cfg = base_cfg or {
        "num_drones": 5, "num_dogs": 0, "max_hops": 1,
        "max_steps": args.max_steps,
    }
    rows: list[pd.DataFrame] = []
    for override in overrides_list:
        label = label_fn(override)
        log.info(f"  --- {label} ---")
        evaluator = SwarmComparativeEvaluator(
            dataset_dir=args.dataset,
            size=args.size,
            num_victims=args.num_victims,
            seeds=seeds,
            budget_per_agent=args.budget,
            swarm_configs=[{**base_cfg, "label": label}],
            **extra_eval_kwargs_fn(override),
        )
        df_o = evaluator.run_all()
        df_o = df_o[df_o["Algorithm"] == label].copy()
        for k, v in override.items():
            df_o[k] = v
        rows.append(df_o)
    return pd.concat(rows, ignore_index=True)


def experiment_6(args: argparse.Namespace) -> pd.DataFrame:
    """Barrido de ``comm_range`` ∈ {250, 500, 1000, 2000, 5000} m.

    Mide cómo afecta el alcance de gossip a la cobertura, solapamiento y
    reparto de carga (Gini E9).
    """
    log.info("=" * 70)
    log.info("  EXPERIMENTO 6: Rango de gossip (comm_range)")
    log.info("=" * 70)

    sweeps = [{"Comm_range": cr} for cr in (250.0, 500.0, 1000.0, 2000.0, 5000.0)]
    df = _sweep_swarm(
        args,
        label_fn=lambda o: f"comm_{int(o['Comm_range'])}",
        overrides_list=sweeps,
        extra_eval_kwargs_fn=lambda o: {"comm_range": o["Comm_range"]},
    )
    csv_path = RESULTS_DIR / "exp6_comm_range.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"  >> Resultados guardados en {csv_path}")
    return df


def plot_experiment_6(df: pd.DataFrame) -> None:
    """Gráfica de evolución con comm_range en eje X (log)."""
    metrics = [
        ("Prob_covered_ratio", "Prob. cubierta (ratio)"),
        ("Overlap_ratio", "Solapamiento (ratio)"),
        ("Victims_pct", "Víctimas encontradas (%)"),
        ("Agent_prob_gini", "Gini reparto por agente (E9)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Experimento 6: Impacto del rango de gossip (comm_range)",
        fontsize=15, fontweight="bold",
    )
    for ax, (col, ylabel) in zip(axes.flat, metrics):
        agg = df.groupby("Comm_range")[col].agg(["mean", "std"]).sort_index()
        ax.errorbar(
            agg.index.astype(float), agg["mean"], yerr=agg["std"],
            marker="o", linewidth=2, capsize=5,
            color="#2196F3", markersize=8,
        )
        ax.set_xscale("log")
        ax.set_xlabel("comm_range (m)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, which="both")
        for x, m in zip(agg.index, agg["mean"]):
            ax.annotate(f"{m:.3f}", (x, m), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9)
    plt.tight_layout()
    out = GRAPHS_DIR / "exp6_comm_range.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  >> Gráfica guardada en {out}")


# ═══════════════════════════════════════════════════════════════════
#  EXPERIMENTO 7: Persistencia de feromonas (evaporation_rate) — docs/20 §E7
# ═══════════════════════════════════════════════════════════════════

def experiment_7(args: argparse.Namespace) -> pd.DataFrame:
    """Barrido de ``evaporation_rate`` ∈ {0.02, 0.01, 0.005, 0.002, 0.001, 0.0}.

    Mantiene ``alert_evaporation_rate`` proporcional (= evap/2). Mide si
    aumentar la vida media de la feromona de exploración reduce el
    re-visitado del centro (problema diagnosticado en docs/18).
    """
    log.info("=" * 70)
    log.info("  EXPERIMENTO 7: Persistencia de feromonas (evaporation_rate)")
    log.info("=" * 70)

    sweeps = [{"Evaporation_rate": e} for e in (0.02, 0.01, 0.005, 0.002, 0.001, 0.0)]
    df = _sweep_swarm(
        args,
        label_fn=lambda o: f"evap_{o['Evaporation_rate']:.4f}",
        overrides_list=sweeps,
        extra_eval_kwargs_fn=lambda o: {
            "evaporation_rate": o["Evaporation_rate"],
            "alert_evaporation_rate": o["Evaporation_rate"] / 2.0,
        },
    )
    csv_path = RESULTS_DIR / "exp7_pheromone_persistence.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"  >> Resultados guardados en {csv_path}")
    return df


def plot_experiment_7(df: pd.DataFrame) -> None:
    """Gráfica de evolución con evaporation_rate (eje X invertido en log)."""
    metrics = [
        ("Prob_covered_ratio", "Prob. cubierta (ratio)"),
        ("Overlap_ratio", "Solapamiento (ratio)"),
        ("Victims_pct", "Víctimas encontradas (%)"),
        ("Agent_prob_gini", "Gini reparto por agente (E9)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Experimento 7: Persistencia de feromonas (evaporation_rate)",
        fontsize=15, fontweight="bold",
    )
    for ax, (col, ylabel) in zip(axes.flat, metrics):
        agg = df.groupby("Evaporation_rate")[col].agg(["mean", "std"]).sort_index()
        x_labels = [f"{e:g}" for e in agg.index]
        ax.errorbar(
            x_labels, agg["mean"], yerr=agg["std"],
            marker="o", linewidth=2, capsize=5,
            color="#4CAF50", markersize=8,
        )
        ax.set_xlabel("evaporation_rate (↘ = más persistencia)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        for x, m in zip(x_labels, agg["mean"]):
            ax.annotate(f"{m:.3f}", (x, m), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9)
    plt.tight_layout()
    out = GRAPHS_DIR / "exp7_pheromone_persistence.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  >> Gráfica guardada en {out}")


# ═══════════════════════════════════════════════════════════════════
#  EXPERIMENTO 8: Hard-mask de celdas observadas — docs/20 §E8
# ═══════════════════════════════════════════════════════════════════

def experiment_8(args: argparse.Namespace) -> pd.DataFrame:
    """Barrido de ``ever_explored_penalty`` ∈ {0.0, 0.25, 0.5, 0.75, 1.0}.

    Con 0.0 reproduce el baseline previo (feromona evaporable como única
    señal de "ya visto"); con 1.0 emula el hard-mask del greedy
    centralizado (re-visitar solo si todas las vecinas también están
    enmascaradas).
    """
    log.info("=" * 70)
    log.info("  EXPERIMENTO 8: Hard-mask sobre celdas observadas")
    log.info("=" * 70)

    sweeps = [{"Ever_explored_penalty": p} for p in (0.0, 0.25, 0.5, 0.75, 1.0)]
    df = _sweep_swarm(
        args,
        label_fn=lambda o: f"eep_{o['Ever_explored_penalty']:.2f}",
        overrides_list=sweeps,
        extra_eval_kwargs_fn=lambda o: {
            "ever_explored_penalty": o["Ever_explored_penalty"],
        },
    )
    csv_path = RESULTS_DIR / "exp8_ever_explored_mask.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"  >> Resultados guardados en {csv_path}")
    return df


def plot_experiment_8(df: pd.DataFrame) -> None:
    """Gráfica de evolución con ever_explored_penalty en eje X."""
    metrics = [
        ("Prob_covered_ratio", "Prob. cubierta (ratio)"),
        ("Overlap_ratio", "Solapamiento (ratio)"),
        ("Victims_pct", "Víctimas encontradas (%)"),
        ("Agent_prob_gini", "Gini reparto por agente (E9)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Experimento 8: Hard-mask sobre celdas observadas (ever_explored_penalty)",
        fontsize=15, fontweight="bold",
    )
    for ax, (col, ylabel) in zip(axes.flat, metrics):
        agg = df.groupby("Ever_explored_penalty")[col].agg(["mean", "std"]).sort_index()
        ax.errorbar(
            agg.index.astype(float), agg["mean"], yerr=agg["std"],
            marker="o", linewidth=2, capsize=5,
            color="#9C27B0", markersize=8,
        )
        ax.set_xlabel("ever_explored_penalty (0=OFF, 1=hard-mask)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        for x, m in zip(agg.index, agg["mean"]):
            ax.annotate(f"{m:.3f}", (x, m), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9)
    plt.tight_layout()
    out = GRAPHS_DIR / "exp8_ever_explored_mask.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  >> Gráfica guardada en {out}")


# ═══════════════════════════════════════════════════════════════════
#  EXPERIMENTO 9: Reparto de carga por agente — docs/20 §E9
# ═══════════════════════════════════════════════════════════════════

def experiment_9(args: argparse.Namespace) -> pd.DataFrame:
    """Estadística de probabilidad acumulada por agente.

    A diferencia de E6–E8 (que son barridos), E9 ejecuta UNA sola
    configuración "buena" del enjambre y extrae las curvas
    ``cumulative_probability_swept`` por agente, junto con el índice de
    Gini sobre el reparto final. Sirve de complemento transversal al
    resto del capítulo 6: la métrica también aparece en los DataFrames
    de E1–E8 vía ``Agent_prob_gini``.
    """
    log.info("=" * 70)
    log.info("  EXPERIMENTO 9: Reparto de carga por agente (probabilidad)")
    log.info("=" * 70)

    seeds = SEED_LIST[:args.seeds]
    swarm_cfg = {
        "num_drones": 3, "num_dogs": 2, "max_hops": 1,
        "max_steps": args.max_steps,
    }
    rows: list[dict] = []
    final_per_agent: list[dict] = []

    for seed in seeds:
        evaluator = SwarmComparativeEvaluator(
            dataset_dir=args.dataset, size=args.size,
            num_victims=args.num_victims, seeds=[seed],
            budget_per_agent=args.budget,
            swarm_configs=[{**swarm_cfg, "label": "Swarm_3D2P"}],
        )
        item, victims_gdf = evaluator._load_scenario(seed)
        if item is None:
            continue
        # Re-ejecutamos manualmente para tener acceso al objeto sim
        # (necesitamos las trayectorias por agente, no solo el resumen).
        from sarenv.swarm.config import SwarmConfig, DroneConfig, RobotDogConfig
        from sarenv.swarm.simulator import SwarmSimulator
        from sarenv.swarm.metrics import SwarmMetrics

        drone_cfg = DroneConfig(altitude=evaluator.altitude, fov_deg=evaluator.fov_deg)
        dog_cfg = RobotDogConfig(sensor_range=20.0)
        for c in (drone_cfg, dog_cfg):
            c.anti_revisit_window = evaluator.anti_revisit_window
            c.anti_revisit_penalty = evaluator.anti_revisit_penalty
            c.presence_weight = evaluator.presence_weight
            c.pheromone_attenuation = evaluator.pheromone_attenuation
            c.dispersal_weight = evaluator.dispersal_weight
        config = SwarmConfig(
            num_drones=swarm_cfg["num_drones"],
            num_dogs=swarm_cfg["num_dogs"],
            budget_per_agent=args.budget,
            max_steps=args.max_steps,
            max_hops=swarm_cfg["max_hops"],
            drone_config=drone_cfg, dog_config=dog_cfg,
            presence_diffusion_sigma=evaluator.presence_diffusion_sigma,
        )

        sim = SwarmSimulator.from_dataset_item(item, config, seed=seed)
        sim.run()
        report = SwarmMetrics(sim, victims=victims_gdf).full_report()

        rows.append({
            "Seed": seed,
            "Algorithm": "Swarm_3D2P",
            "Total_prob_swept": report["total_probability_swept"],
            "Mean_prob_swept": report["mean_probability_swept"],
            "Agent_prob_gini": report["agent_probability_gini"],
            "Coverage_ratio": report["coverage_ratio"],
            "Prob_covered_ratio": report["probability_coverage_ratio"],
        })
        for aid, swept in report["per_agent_probability_swept"].items():
            final_per_agent.append({
                "Seed": seed,
                "Agent_id": aid,
                "Agent_type": "drone" if aid.startswith("drone") else "robot_dog",
                "Cumulative_prob_swept": swept,
                "Cells_explored": report["per_agent_explored"].get(aid, 0),
            })

    df_summary = pd.DataFrame(rows)
    df_per_agent = pd.DataFrame(final_per_agent)

    csv_path = RESULTS_DIR / "exp9_load_balance.csv"
    df_summary.to_csv(csv_path, index=False)
    per_agent_path = RESULTS_DIR / "exp9_load_balance_per_agent.csv"
    df_per_agent.to_csv(per_agent_path, index=False)
    log.info(f"  >> Resumen en {csv_path}")
    log.info(f"  >> Detalle por agente en {per_agent_path}")
    return df_summary


def plot_experiment_9(df: pd.DataFrame) -> None:
    """Gráficas E9: barras por agente, histograma del Gini, ECDF de reparto."""
    per_agent_path = RESULTS_DIR / "exp9_load_balance_per_agent.csv"
    if not per_agent_path.exists():
        log.warning("  No se encontró el CSV per-agent de E9; salto plot.")
        return
    df_pa = pd.read_csv(per_agent_path)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(
        "Experimento 9: Reparto de probabilidad cubierta por agente",
        fontsize=14, fontweight="bold",
    )

    # (a) Probabilidad acumulada media por agente (barras con std entre semillas)
    ax = axes[0]
    agg = df_pa.groupby("Agent_id")["Cumulative_prob_swept"].agg(["mean", "std"])
    agg = agg.sort_index()
    colors = ["#2196F3" if aid.startswith("drone") else "#FF9800"
              for aid in agg.index]
    ax.bar(agg.index, agg["mean"], yerr=agg["std"], capsize=4,
           color=colors, alpha=0.85, edgecolor="black")
    ax.set_title("Probabilidad acumulada por agente (media ± std)")
    ax.set_ylabel("∑ prob (celdas nuevas)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)

    # (b) Distribución del índice de Gini entre semillas
    ax = axes[1]
    ax.hist(df["Agent_prob_gini"].dropna(), bins=10,
            color="#9C27B0", alpha=0.85, edgecolor="black")
    ax.axvline(df["Agent_prob_gini"].mean(), color="red", linestyle="--",
               label=f"media = {df['Agent_prob_gini'].mean():.3f}")
    ax.set_title("Distribución del Gini entre semillas")
    ax.set_xlabel("Gini (0 = reparto perfecto)")
    ax.set_ylabel("Frecuencia")
    ax.legend()
    ax.grid(alpha=0.3)

    # (c) ECDF del reparto: ordenamos agentes y vemos qué fracción
    # acumula qué fracción de masa
    ax = axes[2]
    for seed, group in df_pa.groupby("Seed"):
        vals = np.sort(group["Cumulative_prob_swept"].values)
        total = vals.sum() or 1.0
        cum = np.cumsum(vals) / total
        x = np.linspace(0, 1, len(vals))
        ax.plot(x, cum, marker="o", alpha=0.6, label=f"seed {seed}")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="reparto ideal")
    ax.set_title("Curva de Lorenz (reparto de masa)")
    ax.set_xlabel("Fracción de agentes (ordenados)")
    ax.set_ylabel("Fracción acumulada de prob")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = GRAPHS_DIR / "exp9_load_balance.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  >> Gráfica guardada en {out}")


# ═══════════════════════════════════════════════════════════════════
#  RESUMEN
# ═══════════════════════════════════════════════════════════════════

def generate_summary_table(exp_dfs: dict[int, pd.DataFrame]) -> None:
    """Genera una tabla resumen LaTeX-friendly con los resultados clave."""
    lines: list[str] = []
    lines.append("# Resumen de Resultados — Fase 4\n")

    for exp_num, df in sorted(exp_dfs.items()):
        lines.append(f"\n## Experimento {exp_num}\n")
        # Algunos experimentos (e.g. E9) usan columnas distintas; saltamos
        # la tabla genérica si faltan las columnas esperadas.
        required = {"Algorithm", "Likelihood", "Victims_pct", "Area_km2", "Elapsed_s"}
        if not required.issubset(df.columns):
            available = sorted(c for c in df.columns
                               if df[c].dtype.kind in "fi" and c != "Seed")
            lines.append(
                f"_Resultados en CSV (columnas: {', '.join(available)})._"
            )
            continue
        # Media ± std por Algorithm
        agg = df.groupby("Algorithm").agg(
            Likelihood_mean=("Likelihood", "mean"),
            Likelihood_std=("Likelihood", "std"),
            Victims_mean=("Victims_pct", "mean"),
            Victims_std=("Victims_pct", "std"),
            Area_mean=("Area_km2", "mean"),
            Area_std=("Area_km2", "std"),
            Elapsed_mean=("Elapsed_s", "mean"),
        ).round(3)

        lines.append("| Algorithm | Likelihood | Victims (%) | Area (km²) | Tiempo (s) |")
        lines.append("|-----------|-----------|-------------|-----------|-----------|")
        for algo, row in agg.iterrows():
            l_str = f"{row['Likelihood_mean']:.3f} ± {row['Likelihood_std']:.3f}" if pd.notna(row['Likelihood_std']) else f"{row['Likelihood_mean']:.3f}"
            v_str = f"{row['Victims_mean']:.1f} ± {row['Victims_std']:.1f}" if pd.notna(row['Victims_std']) else f"{row['Victims_mean']:.1f}"
            a_str = f"{row['Area_mean']:.2f} ± {row['Area_std']:.2f}" if pd.notna(row['Area_std']) else f"{row['Area_mean']:.2f}"
            t_str = f"{row['Elapsed_mean']:.1f}"
            lines.append(f"| {algo} | {l_str} | {v_str} | {a_str} | {t_str} |")

    summary_path = RESULTS_DIR / "summary_phase4.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"  >> Tabla resumen guardada en {summary_path}")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    GRAPHS_DIR.mkdir(exist_ok=True)

    t0 = time.perf_counter()
    exp_dfs: dict[int, pd.DataFrame] = {}

    if 1 in args.exp:
        df1 = experiment_1(args)
        exp_dfs[1] = df1
        plot_experiment_1(df1)

    if 2 in args.exp:
        df2 = experiment_2(args)
        exp_dfs[2] = df2
        plot_experiment_2(df2)

    if 3 in args.exp:
        df3 = experiment_3(args)
        exp_dfs[3] = df3
        plot_experiment_3(df3)

    if 4 in args.exp:
        df4 = experiment_4(args)
        exp_dfs[4] = df4
        plot_experiment_4(df4)

    if 5 in args.exp:
        df5 = experiment_5(args)
        exp_dfs[5] = df5
        plot_experiment_5(df5)

    if 6 in args.exp:
        df6 = experiment_6(args)
        exp_dfs[6] = df6
        plot_experiment_6(df6)

    if 7 in args.exp:
        df7 = experiment_7(args)
        exp_dfs[7] = df7
        plot_experiment_7(df7)

    if 8 in args.exp:
        df8 = experiment_8(args)
        exp_dfs[8] = df8
        plot_experiment_8(df8)

    if 9 in args.exp:
        df9 = experiment_9(args)
        exp_dfs[9] = df9
        plot_experiment_9(df9)

    # Tabla resumen
    if exp_dfs:
        generate_summary_table(exp_dfs)

    total = time.perf_counter() - t0
    log.info(f"\n{'=' * 70}")
    log.info(f"  FASE 4 COMPLETADA en {total:.0f}s ({total/60:.1f} min)")
    log.info(f"  Resultados en {RESULTS_DIR}/")
    log.info(f"  Gráficas en {GRAPHS_DIR}/")
    log.info(f"{'=' * 70}")


if __name__ == "__main__":
    main()
