#!/usr/bin/env python
"""
05_comparative_video.py

Comparative coverage video: Swarm (5D+10P) vs Pizza vs Greedy.

Layout
------
·  Swarm map   (top-left)   ·  Pizza map   (top-right)
·  Greedy map  (bottom-left)·  [empty]     (bottom-right)
·  4 stacked graphs on the right column:
     1. Área Cubierta (km²)
     2. Probabilidad Cubierta (%) = likelihood_score / heatmap_sum × 100
     3. Likelihood Score (time-discounted)
     4. Víctimas Encontradas (%)

Usage
-----
    python examples/02_swarm/05_comparative_video.py
    python examples/02_swarm/05_comparative_video.py --num_drones 5 --num_dogs 10 \
        --budget 100000 --interval 2500 --fps 24 --fpi 6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
from shapely.geometry import Point

import sarenv
from sarenv.analytics import metrics as met
from sarenv.analytics.paths import generate_pizza_zigzag_path, generate_greedy_path
from sarenv.core.loading import DatasetLoader
from sarenv.core.lost_person import LostPersonLocationGenerator
from sarenv.swarm import SwarmConfig, DroneConfig, SwarmSimulator

log = sarenv.get_logger()

# ── Colour palette ─────────────────────────────────────────────────────────────

ALGO_COLORS = {
    "Swarm":  "#7B2FBE",   # purple
    "Pizza":  "#E63946",   # red
    "Greedy": "#2D9E57",   # green
}

# Per-agent colours for the Swarm map (drones = blues, dogs = reds/oranges)
_DRONE_COLORS = ["#1f77b4", "#aec7e8", "#6baed6", "#2171b5", "#08519c",
                 "#41b6c4", "#1d91c0", "#225ea8", "#253494", "#081d58"]
_DOG_COLORS   = ["#d62728", "#ff7f0e", "#e6550d", "#fd8d3c", "#e7298a",
                 "#f768a1", "#ae017e", "#7a0177", "#49006a", "#800026"]

# Map positions in GridSpec(8, 6): (row_start, col_start, row_span, col_span)
MAP_GRID = {
    "Swarm":  (0, 0, 4, 2),
    "Pizza":  (0, 2, 4, 2),
    "Greedy": (4, 0, 4, 2),
}

# Graph positions: (row_start, col_start) each spans 2 rows, 2 cols (cols 4-5)
GRAPH_GRID = [
    (0, 4),   # Area Cubierta
    (2, 4),   # Probabilidad Cubierta
    (4, 4),   # Likelihood (time-discounted)
    (6, 4),   # Víctimas Encontradas
]

GRAPH_META = [
    ("area_covered",       "Área Cubierta",           "km²",  False),
    ("likelihood_score",   "Prob. Cubierta",           "%",    True),   # normalise by heatmap_sum
    ("time_discounted_score", "Likelihood Score",      "",     False),
    ("victims_found_pct",  "Víctimas Encontradas",     "%",    False),
]


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Comparative Swarm vs Pizza vs Greedy video")
    p.add_argument("--dataset",    default="maigmo_dataset")
    p.add_argument("--size",       default="medium")
    p.add_argument("--num_drones", type=int,   default=5)
    p.add_argument("--num_dogs",   type=int,   default=10)
    p.add_argument("--budget",     type=float, default=100_000,
                   help="Budget per agent in metres (default: 100000 = 100 km)")
    p.add_argument("--max_steps",  type=int,   default=5_000)
    p.add_argument("--max_hops",   type=int,   default=1)
    p.add_argument("--num_victims",type=int,   default=200)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--interval",   type=float, default=2500.0,
                   help="Metres between metric samples (default: 2500)")
    p.add_argument("--fpi",        type=int,   default=6,
                   help="Frames per interval (smoothness), default 6")
    p.add_argument("--fps",        type=int,   default=24)
    p.add_argument("--dpi",        type=int,   default=90)
    p.add_argument("--output",     default="coverage_videos/swarm_vs_pizza_vs_greedy.mp4")
    return p.parse_args()


# ── Frame renderer ─────────────────────────────────────────────────────────────

def _render_frame(
    item,
    all_data: dict,
    interval_idx: int,
    heatmap_sum: float,
    args: argparse.Namespace,
) -> np.ndarray:
    """Render one video frame and return it as a BGR numpy array."""
    fig = plt.figure(figsize=(22, 12), dpi=args.dpi)
    canvas = FigureCanvasAgg(fig)

    gs = fig.add_gridspec(
        8, 6,
        width_ratios=[1, 1, 1, 1, 1.1, 1.1],
        hspace=0.35,
        wspace=0.28,
    )

    # ── Map axes ───────────────────────────────────────────────────────
    ax_maps: dict[str, plt.Axes] = {}
    for algo, (r, c, rs, cs) in MAP_GRID.items():
        ax_maps[algo] = fig.add_subplot(gs[r:r + rs, c:c + cs])

    # ── Graph axes ─────────────────────────────────────────────────────
    ax_graphs: list[plt.Axes] = []
    for r, c in GRAPH_GRID:
        ax_graphs.append(fig.add_subplot(gs[r:r + 2, c:c + 2]))

    # Compute current cumulative distance in km from any algorithm
    current_km = 0.0
    for data in all_data.values():
        anim = data.get("anim", {})
        dists = anim.get("interval_distances", [])
        if interval_idx < len(dists):
            current_km = dists[interval_idx] / 1000.0
            break

    # ── Render maps ────────────────────────────────────────────────────
    x_min, y_min, x_max, y_max = item.bounds

    for algo_name, ax in ax_maps.items():
        ax.imshow(
            item.heatmap,
            extent=(x_min, x_max, y_min, y_max),
            origin="lower", cmap="YlOrRd", alpha=0.75, zorder=1,
        )
        color = ALGO_COLORS[algo_name]
        ax.set_title(algo_name, color=color, fontsize=11, fontweight="bold", pad=4)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(left=False, bottom=False)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

        if algo_name not in all_data:
            continue

        anim = all_data[algo_name]["anim"]
        agent_colors = all_data[algo_name]["colors"]
        path_coords_all = anim.get("path_coordinates", [])
        positions_all   = anim.get("positions", [])

        # Draw progressive paths for each agent
        for pi, drone_coords_per_interval in enumerate(path_coords_all):
            idx = min(interval_idx, len(drone_coords_per_interval) - 1)
            coords = drone_coords_per_interval[idx]
            if len(coords) < 2:
                continue
            xs, ys = zip(*coords)
            clr = agent_colors[pi % len(agent_colors)]
            ax.plot(xs, ys, linewidth=0.7, alpha=0.80, color=clr, zorder=2)

        # Draw current agent positions
        if interval_idx < len(positions_all):
            for pi, (px, py) in enumerate(positions_all[interval_idx]):
                clr = agent_colors[pi % len(agent_colors)]
                is_dog = (algo_name == "Swarm" and pi >= args.num_drones)
                marker = "s" if is_dog else "o"
                ax.plot(
                    px, py,
                    marker=marker, markersize=6, color=clr,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=5,
                )

    # ── Render graphs ──────────────────────────────────────────────────
    for ax_g, (key, title, ylabel, as_prob) in zip(ax_graphs, GRAPH_META):
        ax_g.set_title(title, fontsize=9, fontweight="bold")
        ax_g.set_xlabel("Distancia (km / agente)", fontsize=7)
        ax_g.set_ylabel(ylabel, fontsize=7)
        ax_g.tick_params(labelsize=7)
        ax_g.grid(True, alpha=0.25, linewidth=0.5)

        for algo_name, data in all_data.items():
            anim = data["anim"]
            m_list  = anim.get("metrics", [])
            dists   = anim.get("interval_distances", [])
            end_idx = min(interval_idx + 1, len(m_list))
            if end_idx == 0:
                continue

            vals     = [m[key] for m in m_list[:end_idx]]
            dist_km  = [d / 1000.0 for d in dists[:end_idx]]

            if as_prob:
                vals = [v / heatmap_sum * 100.0 for v in vals]

            ax_g.plot(
                dist_km, vals,
                color=ALGO_COLORS[algo_name],
                linewidth=1.8,
                label=algo_name,
            )

        ax_g.legend(fontsize=7, loc="upper left", framealpha=0.6)

    # ── Figure title ───────────────────────────────────────────────────
    n_d, n_p = args.num_drones, args.num_dogs
    fig.suptitle(
        f"Swarm ({n_d}D+{n_p}P)  vs  Pizza  vs  Greedy  —  Maigmó {args.size}"
        f"  |  {current_km:.1f} km / agente acumulados",
        fontsize=12, y=1.00,
    )

    # ── Encode to BGR numpy array (fixed size via canvas) ──────────────
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    return bgr


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    n_total      = args.num_drones + args.num_dogs
    total_budget = args.budget * n_total

    # ── Load environment ──────────────────────────────────────────────
    log.info("Cargando entorno %s / %s …", args.dataset, args.size)
    loader = DatasetLoader(dataset_directory=args.dataset)
    item   = loader.load_environment(args.size)
    if item is None:
        log.error("No se pudo cargar el dataset '%s'", args.dataset)
        return

    heatmap_sum = float(item.heatmap.sum())
    log.info("  heatmap %s  sum=%.4f", item.heatmap.shape, heatmap_sum)

    # ── Generate victims ──────────────────────────────────────────────
    vgen        = LostPersonLocationGenerator(item)
    vpts        = vgen.generate_locations(args.num_victims, percent_random_samples=0)
    victims_gdf = gpd.GeoDataFrame(geometry=vpts, crs=vgen.features.crs)
    log.info("  %d víctimas generadas", len(victims_gdf))

    # ── PathEvaluator ─────────────────────────────────────────────────
    meter_per_bin = (item.bounds[2] - item.bounds[0]) / item.heatmap.shape[1]
    path_eval = met.PathEvaluator(
        item.heatmap,
        item.bounds,
        victims_gdf,
        fov_deg=45.0,
        altitude=80.0,
        meters_per_bin=meter_per_bin,
    )

    # ── Centre in projected CRS ───────────────────────────────────────
    center_gdf = gpd.GeoDataFrame(
        geometry=[Point(item.center_point)], crs="EPSG:4326"
    ).to_crs(victims_gdf.crs)
    cx       = center_gdf.geometry.iloc[0].x
    cy       = center_gdf.geometry.iloc[0].y
    radius_m = item.radius_km * 1000.0

    # ── Swarm simulation ──────────────────────────────────────────────
    log.info(
        "Ejecutando enjambre (%dD + %dP, budget=%.0fm, steps=%d, hops=%d) …",
        args.num_drones, args.num_dogs, args.budget, args.max_steps, args.max_hops,
    )
    swarm_cfg = SwarmConfig(
        num_drones=args.num_drones,
        num_dogs=args.num_dogs,
        budget_per_agent=args.budget,
        max_steps=args.max_steps,
        max_hops=args.max_hops,
        drone_config=DroneConfig(altitude=80.0, fov_deg=45.0),
    )
    sim = SwarmSimulator.from_dataset_item(item, swarm_cfg, seed=args.seed)
    sim.run()
    swarm_paths = sim.get_paths()
    log.info("  %d trayectorias del enjambre generadas", len(swarm_paths))

    # ── Pizza paths ───────────────────────────────────────────────────
    log.info("Generando rutas Pizza (n=%d, budget=%.0fm) …", n_total, total_budget)
    pizza_paths = generate_pizza_zigzag_path(
        cx, cy, radius_m,
        num_drones=n_total,
        fov_deg=45.0,
        altitude=80.0,
        overlap=0.1,
        path_point_spacing_m=15.0,
        border_gap_m=50.0,
        budget=total_budget,
    )
    log.info("  %d rutas Pizza", len(pizza_paths))

    # ── Greedy paths ──────────────────────────────────────────────────
    log.info("Generando rutas Greedy (n=%d, budget=%.0fm) …", n_total, total_budget)
    greedy_paths = generate_greedy_path(
        cx, cy,
        num_drones=n_total,
        probability_map=item.heatmap,
        bounds=item.bounds,
        max_radius=radius_m,
        budget=total_budget,
        fov_deg=45.0,
        altitude=80.0,
    )
    log.info("  %d rutas Greedy", len(greedy_paths))

    # ── Compute metrics at intervals ──────────────────────────────────
    all_data: dict = {}
    algo_paths = [
        ("Swarm",  swarm_paths),
        ("Pizza",  pizza_paths),
        ("Greedy", greedy_paths),
    ]
    for algo_name, paths in algo_paths:
        log.info("Calculando métricas a intervalos para %s …", algo_name)
        pre = path_eval.calculate_metrics_at_distance_intervals(
            paths,
            discount_factor=0.999,
            interval_distance=args.interval,
        )
        if algo_name == "Swarm":
            drone_c = (_DRONE_COLORS * 4)[:args.num_drones]
            dog_c   = (_DOG_COLORS   * 4)[:args.num_dogs]
            agent_colors = drone_c + dog_c
        else:
            agent_colors = [ALGO_COLORS[algo_name]] * n_total

        all_data[algo_name] = {
            "paths":  paths,
            "colors": agent_colors,
            "anim": {
                "metrics":            pre["interval_metrics"],
                "interval_distances": pre["interval_distances"],
                "positions":          pre["interval_positions"],
                "path_coordinates":   pre.get("interval_path_coordinates", []),
            },
        }
        n_int = pre["total_intervals"]
        log.info("  %d intervalos, último: %.1f km", n_int, pre["interval_distances"][-1] / 1000)

    # ── Determine frame dimensions ────────────────────────────────────
    log.info("Renderizando frame de prueba para determinar dimensiones …")
    test_frame = _render_frame(item, all_data, 0, heatmap_sum, args)
    fh, fw = test_frame.shape[:2]
    log.info("  Tamaño de frame: %dx%d", fw, fh)

    max_intervals = max(len(d["anim"]["metrics"]) for d in all_data.values())
    total_frames  = max_intervals * args.fpi
    log.info(
        "Renderizando %d intervalos × %d fpi = %d frames a %d fps (%.1f s)",
        max_intervals, args.fpi, total_frames, args.fps, total_frames / args.fps,
    )

    # ── Video writer ──────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (fw, fh))
    if not writer.isOpened():
        log.error("No se pudo abrir VideoWriter en: %s", output_path)
        return

    writer.write(test_frame)
    n_written = 1

    for interval_idx in range(1, max_intervals):
        frame = _render_frame(item, all_data, interval_idx, heatmap_sum, args)
        if frame.shape[:2] != (fh, fw):
            frame = cv2.resize(frame, (fw, fh))
        for _ in range(args.fpi):
            writer.write(frame)
        n_written += args.fpi

        if interval_idx % 5 == 0 or interval_idx == max_intervals - 1:
            pct = interval_idx / max_intervals * 100
            log.info("  Intervalo %d/%d (%.0f%%) — %d frames escritos",
                     interval_idx, max_intervals, pct, n_written)

    writer.release()
    log.info("Vídeo guardado: %s  (%d frames a %d fps)", output_path, n_written, args.fps)

    # ── Print final metric summary ─────────────────────────────────────
    print("\n=== Resumen de métricas finales ===")
    for algo_name, data in all_data.items():
        m = data["anim"]["metrics"][-1]
        prob_cov = m["likelihood_score"] / heatmap_sum * 100.0
        print(
            f"  {algo_name:<8}  área={m['area_covered']:.3f} km²"
            f"  prob={prob_cov:.1f}%"
            f"  likelihood={m['likelihood_score']:.4f}"
            f"  víctimas={m['victims_found_pct']:.1f}%"
        )


if __name__ == "__main__":
    main()
