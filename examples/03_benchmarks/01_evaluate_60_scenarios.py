# examples/03_benchmarks/01_evaluate_60_scenarios.py
"""
Evaluación del enjambre sobre los 60 escenarios del benchmark SAREnv.

Carga cada escenario (sarenv_dataset/1..60) como un SARDatasetItem
independiente y ejecuta la simulación de enjambre con la configuración
por defecto (5 drones, max_hops=1, budget adaptado al tamaño del mapa).

Los 60 escenarios se agrupan en 4 categorías:
  - 1-15:  flat/temperate   (~660×660,  ~10 km radius)
  - 16-30: mountainous/temperate (~1220×1220, ~18 km radius)
  - 31-45: flat/dry         (~873×873,  ~13 km radius)
  - 46-60: mountainous/dry  (~1286×1286, ~19 km radius)

Para cada escenario se ejecutan varias semillas y se recogen métricas
de cobertura, víctimas encontradas, overlap y eficiencia.

Uso:
    python examples/03_benchmarks/01_evaluate_60_scenarios.py                    # Todos (60)
    python examples/03_benchmarks/01_evaluate_60_scenarios.py --scenarios 1 5    # Solo 1-5
    python examples/03_benchmarks/01_evaluate_60_scenarios.py --seeds 3          # 3 semillas
    python examples/03_benchmarks/01_evaluate_60_scenarios.py --budget-factor 1.5  # +50% budget
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sarenv.core.loading import SARDatasetItem
from sarenv.core.lost_person import LostPersonLocationGenerator
from sarenv.swarm.comparative import SwarmComparativeEvaluator
from sarenv.swarm.config import SwarmConfig, DroneConfig, RobotDogConfig
from sarenv.swarm.metrics import SwarmMetrics
from sarenv.swarm.simulator import SwarmSimulator
from sarenv.utils.logging_setup import get_logger

log = get_logger()

RESULTS_DIR = Path("results")
GRAPHS_DIR = Path("graphs")
DATASET_DIR = Path("sarenv_dataset")
SEED_LIST = [42, 123, 456, 789, 2025]

# Budget base (metros) por km² de radio del escenario.
# Se escala cuadráticamente con el radio (proporcional al área)
# para que escenarios grandes reciban budget suficiente.
BASE_BUDGET_PER_KM2_RADIUS = 1_200  # metros budget por km² de radio


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluación del enjambre sobre los 60 escenarios SAREnv"
    )
    p.add_argument(
        "--scenarios",
        type=int,
        nargs=2,
        default=[1, 60],
        metavar=("START", "END"),
        help="Rango de escenarios a evaluar (default: 1 60)",
    )
    p.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Número de semillas aleatorias (default: 3)",
    )
    p.add_argument(
        "--num-drones",
        type=int,
        default=3,
        help="Número de drones (default: 3)",
    )
    p.add_argument(
        "--num-dogs",
        type=int,
        default=2,
        help="Número de perros robot (default: 2)",
    )
    p.add_argument(
        "--max-hops",
        type=int,
        default=1,
        help="Profundidad gossip (default: 1)",
    )
    p.add_argument(
        "--budget-factor",
        type=float,
        default=1.0,
        help="Multiplicador del budget base (default: 1.0)",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=15_000,
        help="Máximo de pasos por simulación (default: 15000)",
    )
    p.add_argument(
        "--num-victims",
        type=int,
        default=200,
        help="Número de víctimas por escenario (default: 200)",
    )
    p.add_argument(
        "--no-baselines",
        action="store_true",
        help="No ejecutar algoritmos centralizados (solo swarm)",
    )
    # E6–E8 (docs/21): overrides paramétricos sobre el SwarmConfig por defecto.
    # Si se dejan a None, se mantienen los valores del config.
    p.add_argument(
        "--comm-range", type=float, default=None,
        help="Override de comm_range (m) para drone+dog. Default: del config (500).",
    )
    p.add_argument(
        "--evaporation-rate", type=float, default=None,
        help="Override de evaporation_rate. Default: del config (0.01).",
    )
    p.add_argument(
        "--alert-evaporation-rate", type=float, default=None,
        help="Override de alert_evaporation_rate. Default: del config (0.005).",
    )
    p.add_argument(
        "--ever-explored-penalty", type=float, default=None,
        help="Override de ever_explored_penalty. Default: del config (0.0).",
    )
    p.add_argument(
        "--tag", type=str, default="",
        help="Sufijo opcional para los CSV/gráficas (e.g. 'best_cfg').",
    )
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
#  Carga de escenarios individuales
# ═══════════════════════════════════════════════════════════════════


def load_scenario_item(scenario_id: int) -> SARDatasetItem | None:
    """Carga un escenario individual como SARDatasetItem.

    Lee features.geojson y heatmap.npy del directorio del escenario
    y construye un SARDatasetItem directamente.
    """
    scenario_dir = DATASET_DIR / str(scenario_id)
    features_path = scenario_dir / "features.geojson"
    heatmap_path = scenario_dir / "heatmap.npy"

    if not features_path.exists() or not heatmap_path.exists():
        log.error(f"Escenario {scenario_id}: archivos no encontrados en {scenario_dir}")
        return None

    try:
        with open(features_path, "r") as f:
            geojson_data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        log.error(
            f"Escenario {scenario_id}: features.geojson no es JSON válido "
            "(¿falta git lfs pull?)"
        )
        return None

    heatmap = np.load(heatmap_path)

    center_point = tuple(geojson_data["center_point"])
    meter_per_bin = geojson_data["meter_per_bin"]
    bounds = tuple(geojson_data["bounds"])
    climate = geojson_data.get("climate", "temperate")
    environment_type = geojson_data.get("environment_type", "flat")
    radius_km = geojson_data.get("radius_km", 10.0)

    # Determinar CRS proyectado (UTM)
    lon, lat = center_point
    zone = int((lon + 180) / 6) + 1
    epsg = f"EPSG:326{zone}" if lat >= 0 else f"EPSG:327{zone}"

    # Cargar features y proyectar
    features_gdf = gpd.GeoDataFrame.from_features(
        geojson_data["features"], crs="EPSG:4326"
    )
    features_proj = features_gdf.to_crs(epsg)

    item = SARDatasetItem(
        size="custom",
        center_point=center_point,
        radius_km=radius_km,
        bounds=bounds,
        features=features_proj,
        heatmap=heatmap,
        environment_climate=climate,
        environment_type=environment_type,
    )

    return item


def get_scenario_group(scenario_id: int) -> str:
    """Devuelve el grupo del escenario (para análisis por categoría)."""
    if 1 <= scenario_id <= 15:
        return "flat_temperate"
    elif 16 <= scenario_id <= 30:
        return "mountain_temperate"
    elif 31 <= scenario_id <= 45:
        return "flat_dry"
    elif 46 <= scenario_id <= 60:
        return "mountain_dry"
    return "unknown"


def compute_budget(item: SARDatasetItem, factor: float = 1.0) -> float:
    """Calcula budget por agente proporcional al área del escenario (radius²)."""
    budget = (item.radius_km ** 2) * BASE_BUDGET_PER_KM2_RADIUS * factor
    return budget


# ═══════════════════════════════════════════════════════════════════
#  Evaluación de un escenario
# ═══════════════════════════════════════════════════════════════════


def evaluate_scenario(
    scenario_id: int,
    item: SARDatasetItem,
    args: argparse.Namespace,
) -> list[dict]:
    """Ejecuta el enjambre (y opcionalmente baselines) sobre un escenario.

    Devuelve una lista de dicts con métricas para cada seed × algoritmo.
    """
    seeds = SEED_LIST[: args.seeds]
    budget = compute_budget(item, args.budget_factor)
    group = get_scenario_group(scenario_id)
    rows: list[dict] = []

    for seed in seeds:
        import random

        random.seed(seed)
        np.random.seed(seed)

        # Generar víctimas
        try:
            victim_gen = LostPersonLocationGenerator(item)
            victim_points = victim_gen.generate_locations(
                args.num_victims, percent_random_samples=0
            )
            victims_gdf = gpd.GeoDataFrame(
                geometry=victim_points, crs=item.features.crs
            )
        except Exception as e:
            log.warning(
                f"Escenario {scenario_id}, seed {seed}: error generando víctimas: {e}"
            )
            victims_gdf = gpd.GeoDataFrame(
                columns=["geometry"], crs=item.features.crs
            )

        # ── Swarm ──
        drone_cfg = DroneConfig(altitude=80.0, fov_deg=45.0)
        dog_cfg = RobotDogConfig(sensor_range=20.0)
        # E6–E8 overrides (docs/21)
        if args.comm_range is not None:
            drone_cfg.comm_range = args.comm_range
            dog_cfg.comm_range = args.comm_range
        if args.ever_explored_penalty is not None:
            drone_cfg.ever_explored_penalty = args.ever_explored_penalty
            dog_cfg.ever_explored_penalty = args.ever_explored_penalty

        config = SwarmConfig(
            num_drones=args.num_drones,
            num_dogs=args.num_dogs,
            budget_per_agent=budget,
            max_steps=args.max_steps,
            max_hops=args.max_hops,
            drone_config=drone_cfg,
            dog_config=dog_cfg,
        )
        if args.evaporation_rate is not None:
            config.evaporation_rate = args.evaporation_rate
        if args.alert_evaporation_rate is not None:
            config.alert_evaporation_rate = args.alert_evaporation_rate

        t0 = time.perf_counter()
        sim = SwarmSimulator.from_dataset_item(item, config, seed=seed)
        sim.run()
        elapsed_swarm = time.perf_counter() - t0

        metrics = SwarmMetrics(sim, victims=victims_gdf)
        report = metrics.full_report()

        try:
            pe = metrics.evaluate_with_path_evaluator()
            victim_pct = pe["victim_detection_metrics"].get("percentage_found", 0)
            area_km2 = pe["area_covered"]
            likelihood = pe["total_likelihood_score"]
            path_km = pe["total_path_length"]
        except Exception as e:
            log.warning(f"Escenario {scenario_id}, seed {seed}: PathEvaluator error: {e}")
            victim_pct = 0
            area_km2 = 0
            likelihood = 0
            path_km = 0

        rows.append(
            {
                "Scenario": scenario_id,
                "Group": group,
                "Environment": item.environment_type,
                "Climate": item.environment_climate,
                "Grid": f"{item.heatmap.shape[0]}x{item.heatmap.shape[1]}",
                "Radius_km": item.radius_km,
                "Budget_m": budget,
                "Algorithm": f"Swarm_{args.num_drones}D{args.num_dogs}P",
                "Seed": seed,
                "n_agents": args.num_drones + args.num_dogs,
                "Victims_pct": round(victim_pct, 2),
                "Area_km2": round(area_km2, 2),
                "Coverage_ratio": round(report["coverage_ratio"], 4),
                "Prob_covered_ratio": round(
                    report["probability_coverage_ratio"], 4
                ),
                "Overlap_ratio": round(report["overlap_ratio"], 4),
                "Efficiency_ratio": round(report["efficiency_ratio"], 4),
                "Likelihood": round(likelihood, 6),
                "Path_km": round(path_km, 2),
                "Elapsed_s": round(elapsed_swarm, 1),
            }
        )

        # ── Baselines (si no se desactivan) ──
        if not args.no_baselines:
            from sarenv.analytics import paths as path_algorithms
            from sarenv.analytics.metrics import PathEvaluator

            n_agents = args.num_drones + args.num_dogs
            center_x = (item.bounds[0] + item.bounds[2]) / 2
            center_y = (item.bounds[1] + item.bounds[3]) / 2
            max_radius = max(
                item.bounds[2] - item.bounds[0],
                item.bounds[3] - item.bounds[1],
            ) / 2

            meters_per_bin = int(
                np.ceil(
                    (item.bounds[2] - item.bounds[0]) / item.heatmap.shape[1]
                )
            )

            baselines = {
                "Greedy": path_algorithms.generate_greedy_path,
                "Pizza": path_algorithms.generate_pizza_zigzag_path,
            }

            for algo_name, algo_func in baselines.items():
                t0 = time.perf_counter()
                try:
                    paths = algo_func(
                        center_x=center_x,
                        center_y=center_y,
                        max_radius=max_radius,
                        n_agents=n_agents,
                        budget=budget,
                        heatmap=item.heatmap,
                        extent=item.bounds,
                    )
                    elapsed_base = time.perf_counter() - t0

                    evaluator = PathEvaluator(
                        heatmap=item.heatmap,
                        extent=item.bounds,
                        victims=victims_gdf,
                        fov_deg=45.0,
                        altitude=80.0,
                        meters_per_bin=meters_per_bin,
                    )
                    pe_b = evaluator.calculate_all_metrics(paths)
                    victim_pct_b = pe_b["victim_detection_metrics"].get(
                        "percentage_found", 0
                    )
                    area_km2_b = pe_b["area_covered"]
                    likelihood_b = pe_b["total_likelihood_score"]
                    path_km_b = pe_b["total_path_length"]
                except Exception as e:
                    log.warning(
                        f"Escenario {scenario_id}, {algo_name}: error: {e}"
                    )
                    elapsed_base = 0
                    victim_pct_b = 0
                    area_km2_b = 0
                    likelihood_b = 0
                    path_km_b = 0

                rows.append(
                    {
                        "Scenario": scenario_id,
                        "Group": group,
                        "Environment": item.environment_type,
                        "Climate": item.environment_climate,
                        "Grid": f"{item.heatmap.shape[0]}x{item.heatmap.shape[1]}",
                        "Radius_km": item.radius_km,
                        "Budget_m": budget,
                        "Algorithm": algo_name,
                        "Seed": seed,
                        "n_agents": n_agents,
                        "Victims_pct": round(victim_pct_b, 2),
                        "Area_km2": round(area_km2_b, 2),
                        "Coverage_ratio": 0,
                        "Prob_covered_ratio": 0,
                        "Overlap_ratio": 0,
                        "Efficiency_ratio": 0,
                        "Likelihood": round(likelihood_b, 6),
                        "Path_km": round(path_km_b, 2),
                        "Elapsed_s": round(elapsed_base, 1),
                    }
                )

    return rows


# ═══════════════════════════════════════════════════════════════════
#  Generación de gráficas
# ═══════════════════════════════════════════════════════════════════


def generate_plots(df: pd.DataFrame, output_dir: Path) -> None:
    """Genera gráficas de resultados agregados."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Víctimas por grupo (barplot)
    fig, ax = plt.subplots(figsize=(12, 6))
    means = (
        df.groupby(["Group", "Algorithm"])["Victims_pct"]
        .mean()
        .unstack(fill_value=0)
    )
    means.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel("Víctimas encontradas (%)")
    ax.set_xlabel("Grupo de escenarios")
    ax.set_title("Víctimas por grupo de escenarios y algoritmo")
    ax.legend(title="Algoritmo", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_ylim(0, 105)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(output_dir / "sarenv60_victims_by_group.png", dpi=150)
    plt.close(fig)
    log.info(f"  >> Gráfica: {output_dir / 'sarenv60_victims_by_group.png'}")

    # 2. Cobertura por escenario (lineplot, solo swarm)
    swarm_df = df[df["Algorithm"].str.startswith("Swarm")]
    scenario_means = swarm_df.groupby("Scenario").agg(
        Victims_mean=("Victims_pct", "mean"),
        Victims_std=("Victims_pct", "std"),
        Coverage_mean=("Coverage_ratio", "mean"),
        Overlap_mean=("Overlap_ratio", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Victims
    ax = axes[0]
    ax.bar(
        scenario_means["Scenario"],
        scenario_means["Victims_mean"],
        yerr=scenario_means["Victims_std"],
        capsize=2,
        alpha=0.7,
        color=[
            "#4C72B0" if s <= 15
            else "#DD8452" if s <= 30
            else "#55A868" if s <= 45
            else "#C44E52"
            for s in scenario_means["Scenario"]
        ],
    )
    ax.set_ylabel("Víctimas (%)")
    ax.set_title("Rendimiento del enjambre por escenario")
    ax.axhline(y=scenario_means["Victims_mean"].mean(), color="k", ls="--", alpha=0.5)

    # Coverage
    ax = axes[1]
    ax.bar(
        scenario_means["Scenario"],
        scenario_means["Coverage_mean"] * 100,
        alpha=0.7,
        color=[
            "#4C72B0" if s <= 15
            else "#DD8452" if s <= 30
            else "#55A868" if s <= 45
            else "#C44E52"
            for s in scenario_means["Scenario"]
        ],
    )
    ax.set_ylabel("Cobertura (%)")

    # Overlap
    ax = axes[2]
    ax.bar(
        scenario_means["Scenario"],
        scenario_means["Overlap_mean"] * 100,
        alpha=0.7,
        color=[
            "#4C72B0" if s <= 15
            else "#DD8452" if s <= 30
            else "#55A868" if s <= 45
            else "#C44E52"
            for s in scenario_means["Scenario"]
        ],
    )
    ax.set_ylabel("Overlap (%)")
    ax.set_xlabel("Escenario")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4C72B0", label="Flat/Temperate (1-15)"),
        Patch(facecolor="#DD8452", label="Mountain/Temperate (16-30)"),
        Patch(facecolor="#55A868", label="Flat/Dry (31-45)"),
        Patch(facecolor="#C44E52", label="Mountain/Dry (46-60)"),
    ]
    axes[0].legend(handles=legend_elements, loc="upper right", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / "sarenv60_per_scenario.png", dpi=150)
    plt.close(fig)
    log.info(f"  >> Gráfica: {output_dir / 'sarenv60_per_scenario.png'}")

    # 3. Swarm vs baselines boxplot por grupo
    if "Greedy" in df["Algorithm"].values or "Pizza" in df["Algorithm"].values:
        fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
        groups = ["flat_temperate", "mountain_temperate", "flat_dry", "mountain_dry"]
        group_labels = [
            "Flat/Temperate\n(1-15)",
            "Mountain/Temperate\n(16-30)",
            "Flat/Dry\n(31-45)",
            "Mountain/Dry\n(46-60)",
        ]

        for ax, grp, label in zip(axes, groups, group_labels):
            grp_df = df[df["Group"] == grp]
            algos = grp_df["Algorithm"].unique()
            data = [grp_df[grp_df["Algorithm"] == a]["Victims_pct"].values for a in algos]
            bp = ax.boxplot(data, labels=algos, patch_artist=True)
            colors = ["#4C72B0", "#DD8452", "#55A868"]
            for patch, color in zip(bp["boxes"], colors[: len(bp["boxes"])]):
                patch.set_facecolor(color)
            ax.set_title(label)
            ax.set_ylabel("Víctimas (%)" if ax == axes[0] else "")
            ax.tick_params(axis="x", rotation=45)

        plt.suptitle("Distribución de víctimas por algoritmo y grupo", fontsize=14)
        plt.tight_layout()
        fig.savefig(output_dir / "sarenv60_boxplot_by_group.png", dpi=150)
        plt.close(fig)
        log.info(f"  >> Gráfica: {output_dir / 'sarenv60_boxplot_by_group.png'}")


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    start, end = args.scenarios
    scenario_ids = list(range(start, end + 1))

    log.info("=" * 70)
    log.info(f"  EVALUACIÓN DE {len(scenario_ids)} ESCENARIOS SAREnv")
    log.info(f"  Config: {args.num_drones}D + {args.num_dogs}P, "
             f"max_hops={args.max_hops}, seeds={args.seeds}")
    log.info(f"  Baselines: {'No' if args.no_baselines else 'Greedy + Pizza'}")
    log.info("=" * 70)

    all_rows: list[dict] = []
    total_t0 = time.perf_counter()

    for i, sid in enumerate(scenario_ids):
        log.info(f"\n{'─' * 60}")
        log.info(
            f"  Escenario {sid}/60 ({i + 1}/{len(scenario_ids)}) — "
            f"Grupo: {get_scenario_group(sid)}"
        )
        log.info(f"{'─' * 60}")

        item = load_scenario_item(sid)
        if item is None:
            log.warning(f"  Saltando escenario {sid}")
            continue

        budget = compute_budget(item, args.budget_factor)
        log.info(
            f"  Grid: {item.heatmap.shape}, radius: {item.radius_km:.1f} km, "
            f"budget: {budget / 1000:.0f} km/agente"
        )

        rows = evaluate_scenario(sid, item, args)
        all_rows.extend(rows)

        # Guardar resultados parciales cada 5 escenarios
        suffix = f"_{args.tag}" if args.tag else ""
        if (i + 1) % 5 == 0 or sid == scenario_ids[-1]:
            df_partial = pd.DataFrame(all_rows)
            partial_path = RESULTS_DIR / f"sarenv60_evaluation_partial{suffix}.csv"
            df_partial.to_csv(partial_path, index=False)
            log.info(f"  >> Resultados parciales guardados ({len(all_rows)} filas)")

    total_elapsed = time.perf_counter() - total_t0

    # ── Resultados finales ──
    suffix = f"_{args.tag}" if args.tag else ""
    df = pd.DataFrame(all_rows)
    csv_path = RESULTS_DIR / f"sarenv60_evaluation{suffix}.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"\n{'=' * 70}")
    log.info(f"  RESULTADOS GUARDADOS: {csv_path}")
    log.info(f"  Total: {len(df)} filas, {total_elapsed / 60:.1f} min")
    log.info(f"{'=' * 70}")

    # ── Resumen por grupo ──
    print("\n" + "=" * 70)
    print("  RESUMEN POR GRUPO")
    print("=" * 70)
    for algo in df["Algorithm"].unique():
        print(f"\n  [{algo}]")
        algo_df = df[df["Algorithm"] == algo]
        summary = algo_df.groupby("Group").agg(
            n=("Victims_pct", "count"),
            Victims_mean=("Victims_pct", "mean"),
            Victims_std=("Victims_pct", "std"),
            Area_mean=("Area_km2", "mean"),
            Coverage_mean=("Coverage_ratio", "mean"),
            Overlap_mean=("Overlap_ratio", "mean"),
        )
        for grp, row in summary.iterrows():
            print(
                f"    {grp:25s}: victims={row['Victims_mean']:5.1f}% ± {row['Victims_std']:4.1f}, "
                f"area={row['Area_mean']:6.1f} km², "
                f"coverage={row['Coverage_mean'] * 100:5.1f}%, "
                f"overlap={row['Overlap_mean'] * 100:4.1f}%"
            )

    # Media global
    print("\n  [MEDIA GLOBAL]")
    for algo in df["Algorithm"].unique():
        algo_df = df[df["Algorithm"] == algo]
        print(
            f"    {algo:20s}: victims={algo_df['Victims_pct'].mean():5.1f}% "
            f"± {algo_df['Victims_pct'].std():4.1f}"
        )

    # ── Gráficas ──
    generate_plots(df, GRAPHS_DIR)

    print(f"\n  Tiempo total: {total_elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
