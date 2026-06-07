#!/usr/bin/env python
"""
04_coverage_video.py

Genera un vídeo MP4 del enjambre cubriendo Maigmó tick a tick:
  - Fondo: mapa de probabilidad (YlOrRd)
  - Overlay azul: celdas exploradas acumuladas
  - Puntos de agentes: drones (círculo) y perros (cuadrado)
  - Barra de progreso + métricas en el título

Usage:
    python examples/02_swarm/04_coverage_video.py
    python examples/02_swarm/04_coverage_video.py --num_drones 5 --num_dogs 10
    python examples/02_swarm/04_coverage_video.py --fps 30 --frame_skip 15
"""
from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import cv2
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import Normalize

import sarenv
from sarenv.core.loading import DatasetLoader
from sarenv.core.lost_person import LostPersonLocationGenerator
from sarenv.swarm import SwarmConfig, DroneConfig, SwarmSimulator, SwarmMetrics

log = sarenv.get_logger()

# ── Paleta de colores ────────────────────────────────────────────────────────
DRONE_COLORS = ["#1f77b4", "#aec7e8", "#6baed6", "#2171b5", "#08519c"]
DOG_COLORS   = ["#d62728", "#ff7f0e", "#e6550d", "#fd8d3c", "#e7298a",
                "#f768a1", "#ae017e", "#7a0177", "#49006a", "#800026"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Swarm coverage video generator")
    p.add_argument("--dataset",     default="maigmo_dataset")
    p.add_argument("--size",        default="medium")
    p.add_argument("--num_drones",  type=int,   default=5)
    p.add_argument("--num_dogs",    type=int,   default=10)
    p.add_argument("--budget",      type=float, default=100_000)
    p.add_argument("--max_steps",   type=int,   default=5_000)
    p.add_argument("--max_hops",    type=int,   default=1)
    p.add_argument("--num_victims", type=int,   default=200)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--fps",         type=int,   default=24,
                   help="Fotogramas por segundo del vídeo (default: 24)")
    p.add_argument("--frame_skip",  type=int,   default=20,
                   help="Ticks entre frames renderizados (default: 20)")
    p.add_argument("--dpi",         type=int,   default=100)
    p.add_argument("--output",      default="coverage_videos/swarm_5d10p_medium.mp4")
    return p.parse_args()


def _mark_circle(mask: np.ndarray, row: int, col: int, radius_cells: int) -> None:
    """Marca como exploradas las celdas dentro del radio dado."""
    rows, cols = mask.shape
    for dr in range(-radius_cells, radius_cells + 1):
        for dc in range(-radius_cells, radius_cells + 1):
            if dr * dr + dc * dc <= radius_cells * radius_cells:
                r2, c2 = row + dr, col + dc
                if 0 <= r2 < rows and 0 <= c2 < cols:
                    mask[r2, c2] = True


def render_frame(
    heatmap_rgba: np.ndarray,
    coverage_mask: np.ndarray,
    agent_positions: dict[str, tuple[int, int]],
    agent_active: dict[str, bool],
    item_bounds: tuple[float, float, float, float],
    grid_shape: tuple[int, int],
    tick: int,
    total_ticks: int,
    args: argparse.Namespace,
    dpi: int = 100,
) -> np.ndarray:
    """Renderiza un frame y devuelve array BGR para OpenCV."""
    x_min, y_min, x_max, y_max = item_bounds
    rows, cols = grid_shape

    fig, ax = plt.subplots(figsize=(10, 9), dpi=dpi)

    # Fondo: heatmap
    ax.imshow(heatmap_rgba, extent=(x_min, x_max, y_min, y_max),
              origin="lower", zorder=1)

    # Overlay: cobertura acumulada (azul semitransparente)
    coverage_rgba = np.zeros((*coverage_mask.shape, 4), dtype=np.float32)
    coverage_rgba[coverage_mask, 0] = 0.2   # R
    coverage_rgba[coverage_mask, 2] = 0.8   # B
    coverage_rgba[coverage_mask, 3] = 0.45  # alpha
    ax.imshow(coverage_rgba, extent=(x_min, x_max, y_min, y_max),
              origin="lower", zorder=2, interpolation="nearest")

    # Agentes
    cell_w = (x_max - x_min) / cols
    cell_h = (y_max - y_min) / rows
    total_agents = args.num_drones + args.num_dogs

    for agent_id, (r, c) in agent_positions.items():
        wx = x_min + (c + 0.5) * cell_w
        wy = y_min + (r + 0.5) * cell_h
        is_drone = agent_id.startswith("drone")
        idx = int(agent_id.split("_")[1])
        color = (DRONE_COLORS[idx % len(DRONE_COLORS)] if is_drone
                 else DOG_COLORS[idx % len(DOG_COLORS)])
        alive = agent_active.get(agent_id, True)
        alpha = 1.0 if alive else 0.3
        marker = "o" if is_drone else "s"
        ms = 10 if is_drone else 9
        ax.plot(wx, wy, marker=marker, color=color, markersize=ms,
                alpha=alpha, zorder=5, markeredgecolor="white",
                markeredgewidth=0.6)

    # Leyenda compacta
    drone_patch = mpatches.Patch(color=DRONE_COLORS[0], label=f"Drones ({args.num_drones})")
    dog_patch   = mpatches.Patch(color=DOG_COLORS[0],   label=f"Perros ({args.num_dogs})")
    cov_patch   = mpatches.Patch(color="#3399cc", alpha=0.6, label="Explorado")
    ax.legend(handles=[drone_patch, dog_patch, cov_patch],
              loc="upper right", fontsize=8, framealpha=0.8)

    # Título con métricas
    pct_cov = coverage_mask.sum() / coverage_mask.size * 100
    active_count = sum(1 for v in agent_active.values() if v)
    progress = tick / max(total_ticks, 1) * 100
    ax.set_title(
        f"Tick {tick:>5d}/{total_ticks}  |  Progreso {progress:.0f}%  |  "
        f"Cobertura {pct_cov:.1f}%  |  Agentes activos {active_count}/{total_agents}",
        fontsize=10, pad=6,
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.tick_params(labelsize=7)

    # Renderizar a buffer numpy
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    arr = np.frombuffer(buf.getvalue(), np.uint8)
    frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame_bgr


def main() -> None:
    args = parse_args()

    # ── Cargar entorno ────────────────────────────────────────────────
    log.info(f"Cargando dataset '{args.dataset}' (size={args.size}) ...")
    loader = DatasetLoader(dataset_directory=args.dataset)
    item = loader.load_environment(args.size)
    if item is None:
        log.error("No se pudo cargar el dataset.")
        return

    rows, cols = item.heatmap.shape
    x_min, y_min, x_max, y_max = item.bounds
    log.info(f"  Heatmap {rows}×{cols}, bounds {item.bounds}")

    # ── Víctimas ──────────────────────────────────────────────────────
    victim_gen = LostPersonLocationGenerator(item)
    victim_points = victim_gen.generate_locations(
        args.num_victims, percent_random_samples=0)
    victims_gdf = gpd.GeoDataFrame(
        geometry=victim_points, crs=victim_gen.features.crs)
    log.info(f"  {len(victims_gdf)} víctimas generadas")

    # ── Configurar enjambre ───────────────────────────────────────────
    config = SwarmConfig(
        num_drones=args.num_drones,
        num_dogs=args.num_dogs,
        budget_per_agent=args.budget,
        max_steps=args.max_steps,
        max_hops=args.max_hops,
        drone_config=DroneConfig(altitude=80.0, fov_deg=45.0),
    )

    # Radio de detección en celdas (cell = 30 m)
    cell_m = (x_max - x_min) / cols
    drone_r_cells = max(1, round(config.drone_config.detection_radius / cell_m))
    dog_r_cells   = max(1, round(config.dog_config.detection_radius   / cell_m))
    log.info(f"  Drone detection radius: {drone_r_cells} celdas | "
             f"Dog: {dog_r_cells} celdas")

    # ── Ejecutar simulación ───────────────────────────────────────────
    log.info("Ejecutando simulación (paso a paso) ...")
    sim = SwarmSimulator.from_dataset_item(item, config, seed=args.seed)
    history: list[dict] = []
    for _ in range(args.max_steps):
        snap = sim.step()
        history.append(snap)
        if not any(a.active for a in sim.agents):
            break
    total_ticks = len(history)
    log.info(f"  Simulación finalizada: {total_ticks} ticks")

    # ── Pre-renderizar fondo heatmap ─────────────────────────────────
    norm = Normalize(vmin=float(item.heatmap.min()),
                     vmax=float(item.heatmap.max()))
    cmap = plt.cm.YlOrRd
    heatmap_rgba = cmap(norm(item.heatmap))  # (rows, cols, 4) float
    heatmap_rgba = (heatmap_rgba * 255).astype(np.uint8)

    # ── Preparar vídeo ────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Dimensiones del primer frame (renderizar dummy para obtener tamaño)
    log.info("Determinando dimensiones del frame ...")
    dummy_mask = np.zeros((rows, cols), dtype=bool)
    dummy_pos  = {snap["positions"].keys().__iter__().__next__(): (rows // 2, cols // 2)
                  for snap in [history[0]]}
    dummy_pos  = history[0]["positions"]
    first_frame = render_frame(
        heatmap_rgba, dummy_mask, dummy_pos,
        history[0]["active"], item.bounds, (rows, cols),
        0, total_ticks, args, dpi=args.dpi,
    )
    fh, fw = first_frame.shape[:2]
    log.info(f"  Frame size: {fw}×{fh}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (fw, fh))
    if not writer.isOpened():
        log.error(f"No se pudo abrir VideoWriter: {output_path}")
        return

    # ── Renderizar frames ─────────────────────────────────────────────
    coverage_mask = np.zeros((rows, cols), dtype=bool)
    n_frames = 0
    log.info(f"Renderizando frames (frame_skip={args.frame_skip}) ...")

    for tick_idx, snap in enumerate(history):
        tick = snap["timestep"]

        # Actualizar cobertura con posiciones de este tick
        for agent_id, (r, c) in snap["positions"].items():
            if not snap["active"].get(agent_id, False):
                continue
            is_drone = agent_id.startswith("drone")
            radius = drone_r_cells if is_drone else dog_r_cells
            _mark_circle(coverage_mask, r, c, radius)

        # Renderizar cada frame_skip ticks
        if tick_idx % args.frame_skip == 0 or tick_idx == len(history) - 1:
            frame = render_frame(
                heatmap_rgba, coverage_mask, snap["positions"],
                snap["active"], item.bounds, (rows, cols),
                tick, total_ticks, args, dpi=args.dpi,
            )
            writer.write(frame)
            n_frames += 1
            if n_frames % 20 == 0:
                pct = tick_idx / len(history) * 100
                log.info(f"  Frame {n_frames} — tick {tick}/{total_ticks} "
                         f"({pct:.0f}%) — coverage "
                         f"{coverage_mask.sum() / coverage_mask.size * 100:.1f}%")

    writer.release()
    log.info(f"Vídeo guardado en: {output_path}  ({n_frames} frames a {args.fps} fps)")

    # ── Métricas finales ──────────────────────────────────────────────
    metrics_obj = SwarmMetrics(sim, victims=victims_gdf)
    summary = metrics_obj.coverage_summary()
    log.info("=== Resumen final ===")
    log.info(f"  Coverage ratio:      {summary['coverage_ratio']:.2%}")
    log.info(f"  Prob covered:        {summary['probability_coverage_ratio']:.2%}")
    log.info(f"  Explored cells:      {summary['explored_cells']} / {summary['total_cells']}")


if __name__ == "__main__":
    main()
