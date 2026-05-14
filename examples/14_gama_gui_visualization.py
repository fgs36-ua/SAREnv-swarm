# examples/14_gama_gui_visualization.py
"""
Servidor para visualización 3D en tiempo real con GAMA Platform GUI.

Este script:
  1. Carga un escenario SAR y prepara la simulación
  2. Exporta el heatmap a gama_model/includes/ (para el terreno 3D)
  3. Inicia un servidor TCP en el puerto 6869
  4. Espera a que GAMA Platform se conecte
  5. Ejecuta la simulación tick a tick, enviando posiciones a GAMA

Prerrequisitos:
    - GAMA Platform 2025 instalado (versión GUI, NO headless)
    - Haber exportado datos estáticos al menos una vez (heatmap.csv)

Flujo de uso:
    1. Ejecutar este script:
       python examples/14_gama_gui_visualization.py --scenario 1

    2. Abrir GAMA Platform (GUI)

    3. En GAMA: File → Import → Existing Projects → seleccionar gama_model/

    4. Abrir gama_model/models/sar_network.gaml

    5. Ejecutar el experimento "sar_gui_network"
       → GAMA se conecta al servidor TCP y empieza a recibir datos
       → Se ve la visualización 3D en tiempo real

    # Con delay para visualización más lenta
    python examples/14_gama_gui_visualization.py --scenario 1 --tick-delay-ms 100

    # Con 3 drones y 2 perros
    python examples/14_gama_gui_visualization.py --scenario 1 --num-drones 3 --num-dogs 2
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np

# Ajustar path para imports desde examples/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sarenv.core.loading import SARDatasetItem
from sarenv.core.lost_person import LostPersonLocationGenerator
from sarenv.swarm.config import SwarmConfig
from sarenv.swarm.environment import SwarmEnvironment
from sarenv.swarm.export import export_scenario_for_gama
from sarenv.swarm.gama_network_server import GamaDisconnected, GamaNetworkServer
from sarenv.swarm.simulator import SwarmSimulator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gama_gui")

DATASET_DIR = Path("sarenv_dataset")
GAMA_MODEL_DIR = Path("gama_model")
GAMA_INCLUDES_DIR = GAMA_MODEL_DIR / "includes"
BASE_BUDGET_PER_KM2_RADIUS = 1_200


# ═══════════════════════════════════════════════════════════════════
#  Carga de escenario
# ═══════════════════════════════════════════════════════════════════

def load_scenario_item(scenario_id: int, dataset_dir: Path | None = None) -> SARDatasetItem | None:
    """Carga un escenario individual como SARDatasetItem.

    Si ``dataset_dir`` apunta a un dataset "plano" (con ``features.geojson``
    en la raíz), ignora ``scenario_id`` y carga directamente desde esa carpeta.
    En caso contrario, carga ``dataset_dir/scenario_id/`` (o ``DATASET_DIR/scenario_id/``).
    """
    base = dataset_dir or DATASET_DIR
    if (base / "features.geojson").exists():
        scenario_dir = base
    else:
        scenario_dir = base / str(scenario_id)
    features_path = scenario_dir / "features.geojson"
    heatmap_path = scenario_dir / "heatmap.npy"

    if not features_path.exists() or not heatmap_path.exists():
        log.error("Escenario %d: archivos no encontrados en %s", scenario_id, scenario_dir)
        return None

    try:
        with open(features_path, "r") as f:
            geojson_data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        log.error("Escenario %d: features.geojson no es JSON válido", scenario_id)
        return None

    heatmap = np.load(heatmap_path)
    center_point = tuple(geojson_data["center_point"])
    bounds = tuple(geojson_data["bounds"])
    climate = geojson_data.get("climate", "temperate")
    environment_type = geojson_data.get("environment_type", "flat")
    radius_km = geojson_data.get("radius_km", 10.0)

    lon, lat = center_point
    zone = int((lon + 180) / 6) + 1
    epsg = f"EPSG:326{zone}" if lat >= 0 else f"EPSG:327{zone}"

    features_gdf = gpd.GeoDataFrame.from_features(
        geojson_data["features"], crs="EPSG:4326"
    )
    features_proj = features_gdf.to_crs(epsg)

    return SARDatasetItem(
        size="custom",
        center_point=center_point,
        radius_km=radius_km,
        bounds=bounds,
        features=features_proj,
        heatmap=heatmap,
        environment_climate=climate,
        environment_type=environment_type,
    )


def compute_budget(item: SARDatasetItem, factor: float = 1.0) -> float:
    """Budget por agente proporcional al radio² del escenario."""
    return (item.radius_km ** 2) * BASE_BUDGET_PER_KM2_RADIUS * factor


def generate_victims(
    item: SARDatasetItem,
    num_victims: int,
    env: SwarmEnvironment,
    seed: int = 42,
) -> set[tuple[int, int]]:
    """Genera víctimas y devuelve sus coordenadas grid."""
    np.random.seed(seed)
    try:
        victim_gen = LostPersonLocationGenerator(item)
        victim_points = victim_gen.generate_locations(
            num_victims, percent_random_samples=0,
        )
        victim_cells = set()
        for pt in victim_points:
            r, c = env.world_to_grid(pt.x, pt.y)
            victim_cells.add((r, c))
        return victim_cells
    except Exception as e:
        log.warning("Error generando víctimas: %s", e)
        return set()


# ═══════════════════════════════════════════════════════════════════
#  Simulación con servidor TCP para GAMA GUI
# ═══════════════════════════════════════════════════════════════════

def run_with_gama_gui(
    item: SARDatasetItem,
    args: argparse.Namespace,
) -> dict:
    """Ejecuta la simulación y sirve datos a GAMA GUI vía TCP."""

    # 1. Preparar entorno y simulador
    env = SwarmEnvironment(item)
    budget = compute_budget(item, args.budget_factor)

    config = SwarmConfig(
        num_drones=args.num_drones,
        num_dogs=args.num_dogs,
        max_hops=args.max_hops,
        budget_per_agent=budget,
        max_steps=args.max_steps,
    )
    # Aplicar anti-revisit corto (rompe oscilaciones A-B-A-B observadas en el
    # comportamiento por defecto: novelty satura al floor durante ~30 ticks).
    for cfg in (config.drone_config, config.dog_config):
        cfg.anti_revisit_window = args.anti_revisit_window
        cfg.anti_revisit_penalty = args.anti_revisit_penalty

    sim = SwarmSimulator(env, config, seed=args.seed)
    log.info(
        "Anti-revisit: window=%d ticks, penalty=%.4f",
        args.anti_revisit_window, args.anti_revisit_penalty,
    )
    log.info(
        "Simulador creado: %d drones + %d dogs, budget=%.0f m, grid=%d×%d",
        config.num_drones, config.num_dogs, budget,
        env.grid.rows, env.grid.cols,
    )

    # 2. Generar víctimas
    victim_cells = generate_victims(item, args.num_victims, env, seed=args.seed)
    log.info("Generadas %d víctimas", len(victim_cells))

    # 3. Exportar datos estáticos (heatmap.csv para terreno 3D en GAMA)
    exported = export_scenario_for_gama(
        dataset_item=item,
        environment=env,
        output_dir=GAMA_INCLUDES_DIR,
        victim_cells=victim_cells,
        heatmap_gamma=args.heatmap_gamma,
    )
    log.info("Datos exportados a %s: %s", GAMA_INCLUDES_DIR, list(exported.keys()))

    # 4. Iniciar servidor TCP
    server = GamaNetworkServer(host=args.server_host, port=args.server_port)
    server.start()

    try:
        # 5. Esperar conexión de GAMA
        print()
        print("=" * 60)
        print("  SERVIDOR LISTO — Abre GAMA Platform (GUI)")
        print(f"  Modelo: gama_model/models/sar_network.gaml")
        print(f"  Experimento: sar_gui_network")
        print(f"  Puerto TCP: {args.server_port}")
        print("=" * 60)
        print()

        if not server.wait_for_gama(timeout=300):
            log.error("GAMA no se conectó. Abortando.")
            return {}

        # 6. Enviar datos de inicialización
        try:
            server.send_init(env, sim.agents, victim_cells)
        except GamaDisconnected:
            log.warning("GAMA desconectado durante init. Abortando.")
            return {}
        log.info("Init enviado a GAMA.")

        # 7. Bucle de simulación tick a tick
        found_so_far: set[tuple[int, int]] = set()
        tick_delay = args.tick_delay_ms / 1000.0 if args.tick_delay_ms > 0 else 0
        pheromone_interval = args.pheromone_interval
        gossip_interval = args.gossip_interval
        links_interval = args.links_interval

        # Trail logging opcional: CSV con tick,agent_id,row,col,active,budget
        trail_log_f = None
        trail_log_writer = None
        if args.trail_log:
            import csv as _csv
            trail_log_path = Path(args.trail_log)
            trail_log_path.parent.mkdir(parents=True, exist_ok=True)
            trail_log_f = open(trail_log_path, "w", newline="", encoding="utf-8")
            trail_log_writer = _csv.writer(trail_log_f)
            trail_log_writer.writerow(["tick", "agent_id", "row", "col", "active", "budget"])
            log.info("Trail logging activado: %s", trail_log_path)

        # Gossip event logging: CSV con tick,agent_a,agent_b
        gossip_log_f = None
        gossip_log_writer = None
        if args.gossip_log:
            import csv as _csv
            gossip_log_path = Path(args.gossip_log)
            gossip_log_path.parent.mkdir(parents=True, exist_ok=True)
            gossip_log_f = open(gossip_log_path, "w", newline="", encoding="utf-8")
            gossip_log_writer = _csv.writer(gossip_log_f)
            gossip_log_writer.writerow(["tick", "agent_a", "agent_b"])
            log.info("Gossip event logging activado: %s", gossip_log_path)

        log.info("Iniciando bucle de simulación (max %d steps)...", args.max_steps)
        t0 = time.time()

        for step_num in range(args.max_steps):
            # Ejecutar un tick del simulador
            snapshot = sim.step()

            # Detectar nuevas víctimas encontradas
            new_found: set[tuple[int, int]] = set()
            for agent in sim.agents:
                if agent.active:
                    found_by_agent = agent.cells_ever_explored & victim_cells
                    new_this_tick = found_by_agent - found_so_far
                    new_found.update(new_this_tick)
            found_so_far.update(new_found)

            # Enviar tick a GAMA
            try:
                server.send_tick(
                    snapshot, sim.agents, env,
                    found_victim_cells=new_found if new_found else None,
                )

                # Actualizar feromonas si toca
                if pheromone_interval > 0 and (step_num + 1) % pheromone_interval == 0:
                    server.send_pheromone(sim.agents, GAMA_INCLUDES_DIR)

                # Enviar gossip field si toca (deprecado, off por defecto)
                if gossip_interval > 0 and (step_num + 1) % gossip_interval == 0:
                    server.send_gossip_field(sim.agents, timestep=sim.timestep)

                # Enviar enlaces de comunicación activos (líneas entre agentes en rango)
                if links_interval > 0 and (step_num + 1) % links_interval == 0:
                    server.send_comm_links(sim.agents)
            except GamaDisconnected:
                log.warning("GAMA desconectado en tick %d. Deteniendo simulación.", step_num + 1)
                break

            # Trail logging
            if trail_log_writer is not None:
                t = sim.timestep
                for agent in sim.agents:
                    pos = snapshot["positions"].get(agent.id, agent.position)
                    bud = snapshot["budgets"].get(agent.id, 0.0)
                    act = snapshot["active"].get(agent.id, False)
                    trail_log_writer.writerow([t, agent.id, pos[0], pos[1], int(act), f"{bud:.1f}"])

            # Gossip event logging
            if gossip_log_writer is not None:
                t = sim.timestep
                for a, b in sim.get_active_comm_pairs():
                    gossip_log_writer.writerow([t, a.id, b.id])

            # Log periódico
            if (step_num + 1) % 100 == 0:
                active_count = sum(1 for a in sim.agents if a.active)
                log.info(
                    "Tick %d/%d — activos: %d, víctimas: %d/%d",
                    step_num + 1, args.max_steps, active_count,
                    len(found_so_far), len(victim_cells),
                )

            # Comprobar fin
            if not any(a.active for a in sim.agents):
                log.info("Todos los agentes inactivos en tick %d.", step_num + 1)
                break

            # Delay opcional
            if tick_delay > 0:
                time.sleep(tick_delay)

        elapsed = time.time() - t0
        pct_found = 100.0 * len(found_so_far) / max(1, len(victim_cells))

        log.info("=" * 60)
        log.info("Simulación completada en %.1f s (%d ticks)", elapsed, sim.timestep)
        log.info("Víctimas encontradas: %d/%d (%.1f%%)",
                 len(found_so_far), len(victim_cells), pct_found)
        log.info("=" * 60)

        # Enviar señal de fin
        try:
            server.send_end()
        except GamaDisconnected:
            pass

        if trail_log_f is not None:
            trail_log_f.close()
            log.info("Trail log cerrado.")

        if gossip_log_f is not None:
            gossip_log_f.close()
            log.info("Gossip log cerrado.")

        return {
            "ticks": sim.timestep,
            "elapsed_s": elapsed,
            "victims_found": len(found_so_far),
            "victims_total": len(victim_cells),
            "pct_found": pct_found,
        }

    finally:
        server.stop()


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Servidor TCP para visualización 3D en GAMA Platform GUI",
    )
    p.add_argument("--scenario", type=int, default=1,
                   help="ID del escenario (1-60, default: 1)")
    p.add_argument("--dataset", type=str, default=None,
                   help="Ruta al dataset (default: sarenv_dataset). Ej: maigmo_dataset")
    p.add_argument("--seed", type=int, default=42,
                   help="Semilla aleatoria (default: 42)")
    p.add_argument("--num-drones", type=int, default=5,
                   help="Número de drones (default: 5)")
    p.add_argument("--num-dogs", type=int, default=0,
                   help="Número de perros robot (default: 0)")
    p.add_argument("--max-hops", type=int, default=1,
                   help="Profundidad gossip (default: 1)")
    p.add_argument("--budget-factor", type=float, default=1.0,
                   help="Multiplicador del budget (default: 1.0)")
    p.add_argument("--max-steps", type=int, default=15_000,
                   help="Máximo de ticks (default: 15000)")
    p.add_argument("--num-victims", type=int, default=200,
                   help="Número de víctimas (default: 200)")

    # Servidor TCP
    p.add_argument("--server-host", type=str, default="localhost",
                   help="Host del servidor TCP (default: localhost)")
    p.add_argument("--server-port", type=int, default=6869,
                   help="Puerto del servidor TCP (default: 6869)")
    p.add_argument("--tick-delay-ms", type=int, default=50,
                   help="Delay en ms entre ticks (default: 50)")
    p.add_argument("--pheromone-interval", type=int, default=25,
                   help="Cada cuántos ticks refrescar exploration_field (default: 25)")
    p.add_argument("--gossip-interval", type=int, default=0,
                   help="[DEPRECADO] Antes mandaba el gossip mesh; ahora se ignora (deja 0).")
    p.add_argument("--links-interval", type=int, default=1,
                   help="Cada cuántos ticks enviar enlaces de comunicación (default: 1, 0=desactivado)")
    p.add_argument("--trail-log", type=str, default=None,
                   help="Si se indica, escribe CSV de trayectorias en esa ruta para análisis offline.")
    p.add_argument("--gossip-log", type=str, default=None,
                   help="Si se indica, escribe CSV de eventos gossip (qué pares se comunicaron cada tick) en esa ruta.")
    p.add_argument("--anti-revisit-window", type=int, default=4,
                   help="Ticks recientes a penalizar para romper oscilaciones A-B-A-B (0=off).")
    p.add_argument("--anti-revisit-penalty", type=float, default=0.05,
                   help="Magnitud de la penalización lineal anti-revisita corta.")
    p.add_argument("--heatmap-gamma", type=float, default=1.0,
                   help="Corrección gamma del heatmap exportado a GAMA. "
                        "<1 realza valores bajos (heatmap más extendido), >1 los apaga. "
                        "Default: 1.0 (sin corrección)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset) if args.dataset else None
    ds_label = args.dataset or f"sarenv_dataset/{args.scenario}"
    log.info("Cargando escenario desde %s...", ds_label)

    item = load_scenario_item(args.scenario, dataset_dir=dataset_dir)
    if item is None:
        log.error("No se pudo cargar el escenario desde %s", ds_label)
        sys.exit(1)

    log.info(
        "Escenario %d: %s/%s, radius=%.1f km, heatmap=%s",
        args.scenario,
        getattr(item, "environment_type", "?"),
        getattr(item, "environment_climate", "?"),
        item.radius_km,
        item.heatmap.shape,
    )

    run_with_gama_gui(item, args)


if __name__ == "__main__":
    main()
