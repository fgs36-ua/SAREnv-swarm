# examples/09_big_mixed_simulation.py
"""
Simulación grande con equipo mixto: 10 drones + 10 perros robot.

Budget alto (500 km/agente) y max_steps elevado para dejar que la
simulación cubra todo el terreno que pueda.  Puede tardar bastante
(10-60 min dependiendo de la máquina).

Uso:
    python examples/09_big_mixed_simulation.py
    python examples/09_big_mixed_simulation.py --budget 300000 --max_steps 20000
    python examples/09_big_mixed_simulation.py --num_drones 5 --num_dogs 5
"""
from __future__ import annotations

import argparse
import time
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
    RobotDogConfig,
    SwarmSimulator,
    SwarmMetrics,
)

log = sarenv.get_logger()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulación grande mixta drones + perros.")
    p.add_argument("--dataset", type=str, default="maigmo_dataset")
    p.add_argument("--size", type=str, default="medium")
    p.add_argument("--num_drones", type=int, default=10)
    p.add_argument("--num_dogs", type=int, default=10)
    p.add_argument("--budget", type=float, default=500_000,
                   help="Budget por agente en metros (default: 500000 = 500km)")
    p.add_argument("--max_steps", type=int, default=50_000,
                   help="Límite máximo de ticks (default: 50000)")
    p.add_argument("--max_hops", type=int, default=3)
    p.add_argument("--num_victims", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_plot", action="store_true")
    p.add_argument("--skip_path_eval", action="store_true",
                   help="Saltar PathEvaluator (ahorra tiempo si solo quieres coverage)")
    return p.parse_args()


def run(args: argparse.Namespace) -> None:
    total_agents = args.num_drones + args.num_dogs
    log.info(f"=== Simulación grande: {args.num_drones} drones + {args.num_dogs} dogs ===")
    log.info(f"    Budget: {args.budget/1000:.0f} km/agente, max_steps: {args.max_steps}")

    # ── 1. Cargar entorno ────────────────────────────────────────────
    loader = DatasetLoader(dataset_directory=args.dataset)
    item = loader.load_environment(args.size)
    if item is None:
        log.error("No se pudo cargar el dataset.")
        return
    log.info(f"  Heatmap: {item.heatmap.shape}, bounds: {item.bounds}")

    # ── 2. Generar víctimas ──────────────────────────────────────────
    victim_gen = LostPersonLocationGenerator(item)
    victim_points = victim_gen.generate_locations(args.num_victims, percent_random_samples=0)
    data_crs = victim_gen.features.crs
    victims_gdf = (
        gpd.GeoDataFrame(geometry=victim_points, crs=data_crs)
        if victim_points
        else gpd.GeoDataFrame(columns=["geometry"], crs=data_crs)
    )
    log.info(f"  Víctimas generadas: {len(victims_gdf)}")

    # ── 3. Configurar enjambre mixto ─────────────────────────────────
    config = SwarmConfig(
        num_drones=args.num_drones,
        num_dogs=args.num_dogs,
        budget_per_agent=args.budget,
        max_steps=args.max_steps,
        max_hops=args.max_hops,
        drone_config=DroneConfig(altitude=80.0, fov_deg=45.0),
        dog_config=RobotDogConfig(sensor_range=20.0),
    )
    log.info(f"  Total agentes: {total_agents} ({args.num_drones}D + {args.num_dogs}P)")

    # ── 4. Ejecutar simulación ───────────────────────────────────────
    log.info("Iniciando simulación ... (puede tardar varios minutos)")
    t0 = time.perf_counter()
    simulator = SwarmSimulator.from_dataset_item(item, config, seed=args.seed)
    history = simulator.run()
    elapsed = time.perf_counter() - t0
    log.info(f"Simulación completada: {len(history)} ticks en {elapsed:.1f}s")

    # ── 5. Métricas de cobertura ─────────────────────────────────────
    swarm_metrics = SwarmMetrics(simulator, victims=victims_gdf)
    summary = swarm_metrics.coverage_summary()

    log.info("=" * 60)
    log.info("=== RESULTADOS ===")
    log.info("=" * 60)
    log.info(f"  Ticks totales:        {summary['total_timesteps']}")
    log.info(f"  Tiempo real:          {elapsed:.1f}s")
    log.info(f"  Celdas exploradas:    {summary['explored_cells']} / {summary['total_cells']}")
    log.info(f"  Cobertura:            {summary['coverage_ratio']:.2%}")
    log.info(f"  Prob. cubierta:       {summary['probability_coverage_ratio']:.2%}")
    log.info(f"  Solapamiento:         {summary['overlap_cells']} celdas ({summary['overlap_ratio']:.2%})")
    log.info("-" * 60)

    # Detalle por agente, separado por tipo
    drones = [(aid, cells) for aid, cells in summary["per_agent_explored"].items() if aid.startswith("drone")]
    dogs   = [(aid, cells) for aid, cells in summary["per_agent_explored"].items() if aid.startswith("dog")]

    budget_consumed = summary.get("budget_consumed_m", {})

    log.info("  --- Drones ---")
    for aid, cells in sorted(drones):
        length_m = summary["paths_lengths_m"].get(aid, 0)
        budg_m = budget_consumed.get(aid, 0)
        log.info(f"    {aid}: {cells} celdas, path={length_m/1000:.1f} km, budget={budg_m/1000:.1f} km")

    if dogs:
        log.info("  --- Perros robot (path=distancia real, budget=con coste terreno) ---")
        for aid, cells in sorted(dogs):
            length_m = summary["paths_lengths_m"].get(aid, 0)
            budg_m = budget_consumed.get(aid, 0)
            log.info(f"    {aid}: {cells} celdas, path={length_m/1000:.1f} km, budget={budg_m/1000:.1f} km")

    drone_cells = sum(c for _, c in drones)
    dog_cells = sum(c for _, c in dogs)
    log.info("-" * 60)
    log.info(f"  Celdas por drones total:  {drone_cells}")
    log.info(f"  Celdas por perros total:  {dog_cells}")

    # ── 6. PathEvaluator (opcional) ──────────────────────────────────
    if not args.skip_path_eval:
        log.info("Ejecutando PathEvaluator (puede tardar) ...")
        t1 = time.perf_counter()
        full_metrics = swarm_metrics.evaluate_with_path_evaluator()
        log.info(f"PathEvaluator completado en {time.perf_counter() - t1:.1f}s")
        log.info("=== PathEvaluator Metrics ===")
        for key, val in full_metrics.items():
            if isinstance(val, float):
                log.info(f"  {key}: {val:.4f}")
            elif isinstance(val, dict):
                log.info(f"  {key}: {val}")
            elif isinstance(val, list) and len(val) > 0 and hasattr(val[0], '__len__'):
                log.info(f"  {key}: [{len(val)} arrays]")
            else:
                log.info(f"  {key}: {val}")

    # ── 7. Guardar resultados a CSV ──────────────────────────────────
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / f"big_sim_{args.num_drones}d_{args.num_dogs}p_b{int(args.budget/1000)}k.csv"

    import csv
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["num_drones", args.num_drones])
        w.writerow(["num_dogs", args.num_dogs])
        w.writerow(["budget_per_agent_m", args.budget])
        w.writerow(["max_steps", args.max_steps])
        w.writerow(["seed", args.seed])
        w.writerow(["ticks", summary["total_timesteps"]])
        w.writerow(["elapsed_seconds", f"{elapsed:.1f}"])
        w.writerow(["explored_cells", summary["explored_cells"]])
        w.writerow(["total_cells", summary["total_cells"]])
        w.writerow(["coverage_ratio", f"{summary['coverage_ratio']:.6f}"])
        w.writerow(["probability_coverage_ratio", f"{summary['probability_coverage_ratio']:.6f}"])
        w.writerow(["overlap_cells", summary["overlap_cells"]])
        w.writerow(["overlap_ratio", f"{summary['overlap_ratio']:.6f}"])
        w.writerow(["drone_cells_total", drone_cells])
        w.writerow(["dog_cells_total", dog_cells])
        for aid, cells in summary["per_agent_explored"].items():
            w.writerow([f"cells_{aid}", cells])
        for aid, length in summary["paths_lengths_m"].items():
            w.writerow([f"path_m_{aid}", f"{length:.1f}"])
    log.info(f"Resultados guardados en {csv_path}")

    # ── 8. Plot ──────────────────────────────────────────────────────
    if not args.no_plot:
        _plot(item, simulator, summary, args, elapsed)


def _plot(item, simulator, summary, args, elapsed) -> None:
    log.info("Generando gráfico ...")
    paths = simulator.get_paths()
    x_min, y_min, x_max, y_max = item.bounds

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.imshow(
        item.heatmap,
        extent=(x_min, x_max, y_min, y_max),
        origin="lower",
        cmap="YlOrRd",
        alpha=0.6,
    )

    # Colores distintos para drones (azules) y perros (verdes)
    drone_cmap = plt.cm.Blues(np.linspace(0.4, 0.9, args.num_drones))
    dog_cmap = plt.cm.Greens(np.linspace(0.4, 0.9, max(args.num_dogs, 1)))

    for i, (path, agent) in enumerate(zip(paths, simulator.agents)):
        if path is None or path.is_empty:
            continue
        xs, ys = path.xy
        is_dog = agent.id.startswith("dog")
        idx = i - args.num_drones if is_dog else i
        color = dog_cmap[idx] if is_dog else drone_cmap[idx]
        cells = summary["per_agent_explored"].get(agent.id, 0)
        ax.plot(
            xs, ys,
            linewidth=0.5 if is_dog else 0.7,
            alpha=0.8,
            color=color,
            label=f"{agent.id} ({cells} celdas)",
            linestyle="-" if not is_dog else "--",
        )

    ax.set_title(
        f"Simulación Mixta — {args.num_drones}D + {args.num_dogs}P | "
        f"Cobertura {summary['coverage_ratio']:.1%} | "
        f"Prob. {summary['probability_coverage_ratio']:.1%} | "
        f"{elapsed:.0f}s",
        fontsize=13,
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc="upper right", fontsize=6, ncol=2)

    out_dir = Path("graphs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"big_sim_{args.num_drones}d_{args.num_dogs}p.png"
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Gráfico guardado en {out_file}")


if __name__ == "__main__":
    run(parse_args())
