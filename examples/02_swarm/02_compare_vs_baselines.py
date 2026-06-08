# examples/02_swarm/02_compare_vs_baselines.py
"""
Compare Swarm vs Baseline Algorithms

Runs the swarm simulator alongside the existing path generators
(Greedy, Spiral, Pizza, Concentric, RandomWalk) on the same dataset,
then summarises metrics side-by-side.

Usage:
    python examples/02_swarm/02_compare_vs_baselines.py
    python examples/02_swarm/02_compare_vs_baselines.py --dataset sarenv_dataset/1 --budget 200000 --num_drones 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point

import sarenv
from sarenv.core.loading import DatasetLoader
from sarenv.core.lost_person import LostPersonLocationGenerator
from sarenv.analytics import metrics as metrics_mod
from sarenv.analytics.evaluator import ComparativeEvaluator
from sarenv.swarm import (
    SwarmConfig,
    DroneConfig,
    SwarmSimulator,
    SwarmMetrics,
)

log = sarenv.get_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare swarm simulator against baseline SAR path algorithms."
    )
    parser.add_argument(
        "--dataset", type=str, default="maigmo_dataset",
        help="Dataset directory (default: maigmo_dataset)",
    )
    parser.add_argument(
        "--size", type=str, default="medium",
        help="Environment size (default: medium)",
    )
    parser.add_argument(
        "--num_drones", type=int, default=3,
        help="Number of drones (default: 3)",
    )
    parser.add_argument(
        "--budget", type=int, default=100_000,
        help="Total budget in metres distributed among drones (default: 100000)",
    )
    parser.add_argument(
        "--num_victims", type=int, default=100,
        help="Number of lost persons (default: 100)",
    )
    parser.add_argument(
        "--max_hops", type=int, default=3,
        help="Swarm gossip trust depth (default: 3)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


def run_swarm(item, config, seed) -> SwarmSimulator:
    """Run the swarm simulator and return the completed SwarmSimulator."""
    simulator = SwarmSimulator.from_dataset_item(item, config, seed=seed)
    simulator.run()
    return simulator


def run_baselines(args) -> pd.DataFrame:
    """
    Run existing baseline algorithms through ComparativeEvaluator
    and return the results DataFrame.
    """
    evaluator = ComparativeEvaluator(
        dataset_directory=args.dataset,
        evaluation_sizes=[args.size],
        num_drones=args.num_drones,
        num_lost_persons=args.num_victims,
        budget=args.budget,
    )
    baseline_results, _ = evaluator.run_baseline_evaluations()
    return baseline_results


def build_comparison_table(
    baseline_df: pd.DataFrame,
    swarm_evaluator_metrics: dict,
    swarm_summary: dict,
    size: str,
) -> pd.DataFrame:
    """
    Merge swarm metrics and baseline metrics into a single DataFrame
    for easy comparison.
    """
    # Filter baselines for the requested size
    df_base = baseline_df[baseline_df["Dataset"] == size].copy()

    # Build a swarm row with the same column schema
    swarm_row = {
        "Dataset": size,
        "Algorithm": "Swarm",
        "Likelihood Score": swarm_evaluator_metrics.get("total_likelihood_score", 0),
        "Time-Discounted Score": swarm_evaluator_metrics.get("total_time_discounted_score", 0),
        "Victims Found (%)": swarm_evaluator_metrics.get("victim_detection_metrics", {}).get("percentage_found", 0),
        "Area Covered (km²)": swarm_evaluator_metrics.get("area_covered", 0),
        "Total Path Length (km)": swarm_evaluator_metrics.get("total_path_length", 0),
        # Swarm-specific extras
        "Coverage Ratio (knowledge)": swarm_summary["coverage_ratio"],
        "Overlap Ratio": swarm_summary["overlap_ratio"],
        "Prob Coverage Ratio": swarm_summary["probability_coverage_ratio"],
    }
    swarm_df = pd.DataFrame([swarm_row])

    df = pd.concat([df_base, swarm_df], ignore_index=True)
    return df


def plot_comparison(df: pd.DataFrame, args) -> None:
    """Bar-chart comparing key metrics across algorithms."""
    # Pick the most interesting numeric columns
    interesting = [
        "Likelihood Score",
        "Victims Found (%)",
        "Area Covered (km²)",
        "Total Path Length (km)",
    ]
    cols_to_plot = [c for c in interesting if c in df.columns]
    if not cols_to_plot:
        # Fallback: any numeric column
        cols_to_plot = [
            c for c in df.columns
            if c not in ("Algorithm", "Dataset", "Environment Type", "Climate", "Environment Size")
            and pd.api.types.is_numeric_dtype(df[c])
        ][:5]

    n = len(cols_to_plot)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), squeeze=False)
    axes = axes.flatten()

    algorithms = df["Algorithm"].tolist()
    x = np.arange(len(algorithms))

    for ax, col in zip(axes, cols_to_plot):
        vals = df[col].fillna(0).values
        bars = ax.bar(x, vals, color=plt.cm.Set2.colors[: len(x)])
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, rotation=40, ha="right", fontsize=8)
        ax.set_title(col, fontsize=10)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.3f}" if isinstance(v, float) else str(v),
                ha="center", va="bottom", fontsize=7,
            )

    fig.suptitle(
        f"Swarm vs Baselines — {args.num_drones} drones, budget={args.budget}m",
        fontsize=13,
    )
    fig.tight_layout()

    output_dir = Path("graphs")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"swarm_comparison_n{args.num_drones}_b{args.budget}.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved comparison plot to {out_file}")


def main() -> None:
    args = parse_args()

    # ── 1. Load dataset ──────────────────────────────────────────────
    log.info(f"Loading dataset '{args.dataset}' (size={args.size}) ...")
    loader = DatasetLoader(dataset_directory=args.dataset)
    item = loader.load_environment(args.size)
    if item is None:
        log.error("Failed to load dataset.")
        return

    # ── 2. Generate victims ──────────────────────────────────────────
    victim_gen = LostPersonLocationGenerator(item)
    victim_points = victim_gen.generate_locations(args.num_victims, percent_random_samples=0)
    data_crs = victim_gen.features.crs
    victims_gdf = (
        gpd.GeoDataFrame(geometry=victim_points, crs=data_crs)
        if victim_points
        else gpd.GeoDataFrame(columns=["geometry"], crs=data_crs)
    )
    log.info(f"Generated {len(victims_gdf)} victims")

    # ── 3. Run swarm simulation ──────────────────────────────────────
    budget_per_agent = args.budget / args.num_drones
    swarm_config = SwarmConfig(
        num_drones=args.num_drones,
        budget_per_agent=budget_per_agent,
        max_hops=args.max_hops,
        drone_config=DroneConfig(altitude=80.0, fov_deg=45.0),
    )
    log.info(
        f"Running swarm: {args.num_drones} drones × {budget_per_agent:.0f}m, "
        f"max_hops={args.max_hops}"
    )
    simulator = run_swarm(item, swarm_config, seed=args.seed)
    swarm_metrics = SwarmMetrics(simulator, victims=victims_gdf)
    swarm_summary = swarm_metrics.coverage_summary()
    swarm_evaluator_metrics = swarm_metrics.evaluate_with_path_evaluator()

    log.info(
        f"Swarm done: coverage={swarm_summary['coverage_ratio']:.2%}, "
        f"overlap={swarm_summary['overlap_ratio']:.2%}, "
        f"timesteps={swarm_summary['total_timesteps']}"
    )

    # ── 4. Run baseline algorithms ───────────────────────────────────
    log.info("Running baseline evaluations ...")
    baseline_results = run_baselines(args)
    log.info(f"Baselines done: {len(baseline_results)} algorithm×size combos")

    # ── 5. Build comparison table ────────────────────────────────────
    df = build_comparison_table(
        baseline_results, swarm_evaluator_metrics, swarm_summary, args.size,
    )
    log.info("=== Comparison Table ===")
    log.info("\n" + df.to_string(index=False))

    # Save to CSV
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"swarm_comparison_n{args.num_drones}_b{args.budget}.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"Saved CSV to {csv_path}")

    # ── 6. Generate plot ─────────────────────────────────────────────
    plot_comparison(df, args)

    log.info("Done!")


if __name__ == "__main__":
    main()
