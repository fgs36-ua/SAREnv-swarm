# examples/07_run_swarm_simulation.py
"""
Basic Swarm Simulator Example

Loads a dataset, runs the tick-based swarm simulation, and prints
coverage metrics.  Optionally plots per-agent paths on the heatmap.

Usage:
    python examples/07_run_swarm_simulation.py
    python examples/07_run_swarm_simulation.py --dataset sarenv_dataset/1 --num_drones 5 --budget 150000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sarenv
from sarenv.core.loading import DatasetLoader
from sarenv.core.lost_person import LostPersonLocationGenerator
from sarenv.swarm import (
    SwarmConfig,
    DroneConfig,
    SwarmSimulator,
    SwarmMetrics,
)

log = sarenv.get_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a swarm SAR simulation.")
    parser.add_argument(
        "--dataset", type=str, default="maigmo_dataset",
        help="Path to the dataset directory (default: maigmo_dataset)",
    )
    parser.add_argument(
        "--size", type=str, default="medium",
        help="Environment size to load: small, medium, large, xlarge (default: medium)",
    )
    parser.add_argument(
        "--num_drones", type=int, default=3,
        help="Number of drones in the swarm (default: 3)",
    )
    parser.add_argument(
        "--budget", type=float, default=100_000,
        help="Movement budget per agent in metres (default: 100000)",
    )
    parser.add_argument(
        "--max_steps", type=int, default=5_000,
        help="Maximum simulation ticks (default: 5000)",
    )
    parser.add_argument(
        "--max_hops", type=int, default=3,
        help="Gossip trust depth (default: 3, use 999 for near-centralised)",
    )
    parser.add_argument(
        "--num_dogs", type=int, default=0,
        help="Number of dog robots in the swarm (default: 0)",
    )
    parser.add_argument(
        "--num_victims", type=int, default=100,
        help="Number of lost persons to generate for evaluation (default: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--no_plot", action="store_true",
        help="Skip generating the heatmap plot",
    )
    return parser.parse_args()


def run_simulation(args: argparse.Namespace) -> None:
    # ── 1. Load environment ──────────────────────────────────────────
    log.info(f"Loading dataset '{args.dataset}' (size={args.size}) ...")
    loader = DatasetLoader(dataset_directory=args.dataset)
    item = loader.load_environment(args.size)
    if item is None:
        log.error("Failed to load dataset. Check --dataset and --size flags.")
        return

    log.info(
        f"  heatmap shape: {item.heatmap.shape}, "
        f"  bounds: {item.bounds}, "
        f"  center: {item.center_point}"
    )

    # ── 2. Generate victim locations (for metrics) ───────────────────
    log.info(f"Generating {args.num_victims} victim locations ...")
    victim_gen = LostPersonLocationGenerator(item)
    victim_points = victim_gen.generate_locations(args.num_victims, percent_random_samples=0)
    data_crs = victim_gen.features.crs
    victims_gdf = (
        gpd.GeoDataFrame(geometry=victim_points, crs=data_crs)
        if victim_points
        else gpd.GeoDataFrame(columns=["geometry"], crs=data_crs)
    )
    log.info(f"  generated {len(victims_gdf)} victims")

    # ── 3. Configure swarm ───────────────────────────────────────────
    config = SwarmConfig(
        num_drones=args.num_drones,
        num_dogs=args.num_dogs,
        budget_per_agent=args.budget,
        max_steps=args.max_steps,
        max_hops=args.max_hops,
        drone_config=DroneConfig(
            altitude=80.0,
            fov_deg=45.0,
        ),
    )
    log.info(
        f"Swarm config: {config.num_drones} drones, "
        f"budget={config.budget_per_agent:.0f}m/agent, "
        f"max_hops={config.max_hops}, max_steps={config.max_steps}"
    )

    # ── 4. Run simulation ────────────────────────────────────────────
    log.info("Starting swarm simulation ...")
    simulator = SwarmSimulator.from_dataset_item(item, config, seed=args.seed)
    history = simulator.run()
    log.info(f"Simulation finished: {len(history)} ticks")

    # ── 5. Gather metrics ────────────────────────────────────────────
    swarm_metrics = SwarmMetrics(simulator, victims=victims_gdf)

    # Quick coverage summary (no PathEvaluator)
    summary = swarm_metrics.coverage_summary()
    log.info("=== Coverage Summary ===")
    log.info(f"  Explored cells:       {summary['explored_cells']} / {summary['total_cells']}")
    log.info(f"  Coverage ratio:       {summary['coverage_ratio']:.2%}")
    log.info(f"  Overlap cells:        {summary['overlap_cells']}  (overlap ratio {summary['overlap_ratio']:.2%})")
    log.info(f"  Probability covered:  {summary['probability_coverage_ratio']:.2%}")
    log.info(f"  Total timesteps:      {summary['total_timesteps']}")
    for agent_id, length in summary["paths_lengths_m"].items():
        explored = summary["per_agent_explored"].get(agent_id, 0)
        log.info(f"    {agent_id}: path={length:.0f}m, cells explored={explored}")

    # Full PathEvaluator metrics (compatible with existing evaluation)
    log.info("Running PathEvaluator metrics ...")
    full_metrics = swarm_metrics.evaluate_with_path_evaluator()
    log.info("=== PathEvaluator Metrics ===")
    for key, val in full_metrics.items():
        if isinstance(val, float):
            log.info(f"  {key}: {val:.4f}")
        else:
            log.info(f"  {key}: {val}")

    # ── 6. Plot paths on heatmap ─────────────────────────────────────
    if not args.no_plot:
        _plot_swarm_paths(item, simulator, summary, args)


def _plot_swarm_paths(item, simulator, summary, args) -> None:
    """Plot the heatmap with each agent's path overlaid."""
    log.info("Generating heatmap plot ...")

    paths = simulator.get_paths()
    x_min, y_min, x_max, y_max = item.bounds

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(
        item.heatmap,
        extent=(x_min, x_max, y_min, y_max),
        origin="lower",
        cmap="YlOrRd",
        alpha=0.7,
    )

    colors = plt.cm.tab10.colors
    for i, path in enumerate(paths):
        if path is None or path.is_empty:
            continue
        xs, ys = path.xy
        ax.plot(
            xs, ys,
            linewidth=0.8,
            alpha=0.85,
            color=colors[i % len(colors)],
            label=f"{simulator.agents[i].id} ({summary['paths_lengths_m'][simulator.agents[i].id]:.0f}m)",
        )

    ax.set_title(
        f"Swarm Simulation — {args.num_drones} drones + {args.num_dogs} perros, "
        f"coverage {summary['coverage_ratio']:.1%}, "
        f"hops={args.max_hops}",
        fontsize=12,
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc="upper right", fontsize=8)

    output_dir = Path("graphs")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"swarm_{args.num_drones}d{args.num_dogs}p_hops{args.max_hops}_{args.size}.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved plot to {out_file}")


if __name__ == "__main__":
    run_simulation(parse_args())
