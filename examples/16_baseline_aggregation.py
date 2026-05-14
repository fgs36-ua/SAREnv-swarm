"""Baseline de aglomeración (Iteración 1, ver docs/16).

Ejecuta los 3 escenarios definidos en docs/16 §5 y reporta las métricas
nuevas (`coverage_gini`, `cluster_ratio`, `mean_pair_distance_cells`)
junto con las clásicas, sin modificar la configuración por defecto del
enjambre. Sólo observa.

Salida:
    results/baseline_aggregation_iter1.csv
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sarenv.core.loading import DatasetLoader, SARDatasetItem  # noqa: E402
from sarenv.swarm.config import (  # noqa: E402
    DroneConfig,
    RobotDogConfig,
    SwarmConfig,
)
from sarenv.swarm.metrics import SwarmMetrics  # noqa: E402
from sarenv.swarm.simulator import SwarmSimulator  # noqa: E402

try:
    from sarenv.utils.victims import LostPersonLocationGenerator  # type: ignore
except Exception:
    LostPersonLocationGenerator = None  # type: ignore


SARENV_DIR = ROOT / "sarenv_dataset"


def load_sarenv_scenario(scenario_id: int) -> SARDatasetItem | None:
    """Carga un escenario numerado de sarenv_dataset/<id>/."""
    scen = SARENV_DIR / str(scenario_id)
    feats = scen / "features.geojson"
    hm = scen / "heatmap.npy"
    if not feats.exists() or not hm.exists():
        print(f"  [WARN] escenario {scenario_id} no encontrado en {scen}")
        return None
    with open(feats, "r") as f:
        gj = json.load(f)
    heatmap = np.load(hm)
    lon, lat = gj["center_point"]
    zone = int((lon + 180) / 6) + 1
    epsg = f"EPSG:326{zone}" if lat >= 0 else f"EPSG:327{zone}"
    feats_gdf = gpd.GeoDataFrame.from_features(gj["features"], crs="EPSG:4326")
    feats_proj = feats_gdf.to_crs(epsg)
    return SARDatasetItem(
        size="custom",
        center_point=tuple(gj["center_point"]),
        radius_km=gj.get("radius_km", 10.0),
        bounds=tuple(gj["bounds"]),
        features=feats_proj,
        heatmap=heatmap,
        environment_climate=gj.get("climate", "temperate"),
        environment_type=gj.get("environment_type", "flat"),
    )


def load_maigmo() -> SARDatasetItem | None:
    """Carga el dataset maigmo via DatasetLoader."""
    try:
        loader = DatasetLoader(dataset_directory=str(ROOT / "maigmo_dataset"))
        return loader.load_environment("medium")
    except Exception as exc:
        print(f"  [WARN] maigmo no se pudo cargar: {exc}")
        return None


def gen_victims(item: SARDatasetItem, n: int, seed: int) -> gpd.GeoDataFrame:
    """Genera n víctimas con LostPersonLocationGenerator si está disponible."""
    if LostPersonLocationGenerator is None or n <= 0:
        return gpd.GeoDataFrame(geometry=[], crs=item.features.crs)
    try:
        gen = LostPersonLocationGenerator(item)
        # algunos generadores aceptan seed; intentamos sin romper si no.
        try:
            pts = gen.generate_locations(n, percent_random_samples=0, seed=seed)
        except TypeError:
            pts = gen.generate_locations(n, percent_random_samples=0)
        return gpd.GeoDataFrame(geometry=pts, crs=item.features.crs)
    except Exception as exc:
        print(f"  [WARN] no se pudieron generar víctimas: {exc}")
        return gpd.GeoDataFrame(geometry=[], crs=item.features.crs)


def _presence_weight_for_tag(tag: str) -> float:
    """Mapea tag CLI → presence_weight (Iter3, docs/16).

    Ejemplos:
      iter1            → 0.0  (baseline, OFF)
      iter2            → 0.0  (Reynolds-only intento, OFF)
      iter3            → 0.05 (default Iter3)
      iter3_w0.01      → 0.01
      iter3_w0.1       → 0.10
      iter3_w0.5       → 0.50
    """
    if tag.startswith("iter3"):
        if "_w" in tag:
            try:
                return float(tag.split("_w", 1)[1])
            except ValueError:
                pass
        return 0.05
    return 0.0


def run_one(
    *,
    label: str,
    item: SARDatasetItem,
    num_drones: int,
    num_dogs: int,
    budget: float,
    max_steps: int,
    seed: int,
    num_victims: int,
    presence_weight: float = 0.0,
) -> dict:
    print(f"  [run] {label}: drones={num_drones} dogs={num_dogs} "
          f"budget={budget:.0f} max_steps={max_steps} seed={seed} "
          f"w_presence={presence_weight}")
    cfg = SwarmConfig(
        num_drones=num_drones,
        num_dogs=num_dogs,
        budget_per_agent=budget,
        max_steps=max_steps,
        drone_config=DroneConfig(
            altitude=80.0, fov_deg=45.0, presence_weight=presence_weight,
        ),
        dog_config=RobotDogConfig(
            sensor_range=20.0, presence_weight=presence_weight,
        ),
    )
    victims = gen_victims(item, num_victims, seed)
    t0 = time.perf_counter()
    sim = SwarmSimulator.from_dataset_item(item, cfg, seed=seed)
    sim.run()
    elapsed = time.perf_counter() - t0
    metrics = SwarmMetrics(sim, victims=victims)
    rep = metrics.full_report()
    rep["scenario"] = label
    rep["num_drones"] = num_drones
    rep["num_dogs"] = num_dogs
    rep["budget"] = budget
    rep["seed"] = seed
    rep["elapsed_s"] = elapsed
    print(
        f"    coverage={rep.get('coverage_ratio', 0):.3f} "
        f"prob_cov={rep.get('probability_coverage_ratio', 0):.3f} "
        f"gini={rep.get('coverage_gini', 0):.3f} "
        f"cluster={rep.get('cluster_ratio', 0):.3f} "
        f"meanD={rep.get('mean_pair_distance_cells', 0):.1f} "
        f"eff={rep.get('efficiency_ratio', 0):.3f} "
        f"({elapsed:.1f}s)"
    )
    return rep


def main() -> None:
    seed = 42
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    tag = sys.argv[1] if len(sys.argv) > 1 else "iter1"
    out_csv = out_dir / f"baseline_aggregation_{tag}.csv"
    presence_weight = _presence_weight_for_tag(tag)
    print(f"[cfg] tag={tag}  presence_weight={presence_weight}")

    runs: list[dict] = []

    print("\n=== Escenario 1: sarenv / scenario 1 / 5 drones / budget 100k ===")
    item = load_sarenv_scenario(1)
    if item is not None:
        runs.append(
            run_one(
                label="sarenv_s1_5d_100k",
                item=item,
                num_drones=5,
                num_dogs=0,
                budget=100_000.0,
                max_steps=15_000,
                seed=seed,
                num_victims=20,
                presence_weight=presence_weight,
            )
        )

    print("\n=== Escenario 2: maigmo / 5 drones / budget 250k ===")
    item = load_maigmo()
    if item is not None:
        runs.append(
            run_one(
                label="maigmo_5d_250k",
                item=item,
                num_drones=5,
                num_dogs=0,
                budget=250_000.0,
                max_steps=30_000,
                seed=seed,
                num_victims=20,
                presence_weight=presence_weight,
            )
        )

    print("\n=== Escenario 3: sarenv / scenario 5 / 5 drones + 2 perros / budget 300k ===")
    item = load_sarenv_scenario(5)
    if item is not None:
        runs.append(
            run_one(
                label="sarenv_s5_5d2p_300k",
                item=item,
                num_drones=5,
                num_dogs=2,
                budget=300_000.0,
                max_steps=30_000,
                seed=seed,
                num_victims=20,
                presence_weight=presence_weight,
            )
        )

    if not runs:
        print("Sin runs ejecutados.")
        return

    # Unión de claves (las métricas devuelven el mismo schema, pero
    # nos protegemos por si alguna falta).
    fieldnames: list[str] = []
    seen: set[str] = set()
    preferred = [
        "scenario", "num_drones", "num_dogs", "budget", "seed",
        "coverage_ratio", "probability_coverage_ratio",
        "coverage_gini", "cluster_ratio", "mean_pair_distance_cells",
        "overlap_ratio", "efficiency_ratio",
        "effective_cells", "redundant_visits",
        "time_to_first_victim", "info_propagation_latency",
        "elapsed_s",
    ]
    for k in preferred:
        if k not in seen:
            fieldnames.append(k)
            seen.add(k)
    for r in runs:
        for k in r.keys():
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            w.writerow(r)
    print(f"\n[OK] CSV escrito en {out_csv}")

    # Imprimir tabla resumen para copiar a docs/16
    print("\n=== Resumen para docs/16 §5 ===")
    print(
        "| Escenario | coverage_ratio | coverage_gini | cluster_ratio | "
        "mean_pair_dist | efficiency_ratio |"
    )
    print("|---|---|---|---|---|---|")
    for r in runs:
        print(
            f"| {r['scenario']} | "
            f"{r.get('coverage_ratio', 0):.3f} | "
            f"{r.get('coverage_gini', 0):.3f} | "
            f"{r.get('cluster_ratio', 0):.3f} | "
            f"{r.get('mean_pair_distance_cells', 0):.1f} | "
            f"{r.get('efficiency_ratio', 0):.3f} |"
        )


if __name__ == "__main__":
    main()
