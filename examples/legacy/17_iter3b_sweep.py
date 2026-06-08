"""Iter3.b — Barrido de hiperparámetros de la feromona de presencia.

Reutiliza la infraestructura de `examples/16_baseline_aggregation.py`
para barrer los dos hiperparámetros clave introducidos en Iteración 3
(docs/16 §6.3):

  * `presence_weight`        — peso del término -w*presence_field en el
                               scoring del agente.
  * `presence_evaporation`   — tasa de decaimiento del campo por tick.

Modos de ejecución:

  python examples/17_iter3b_sweep.py weight    # barrido en sarenv_s1
  python examples/17_iter3b_sweep.py evap      # barrido en maigmo

Salida:
  results/iter3b_sweep_weight.csv    (modo 'weight')
  results/iter3b_sweep_evap.csv      (modo 'evap')
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Reusar helpers del ejemplo 16 sin duplicar código
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "ex16", ROOT / "examples" / "16_baseline_aggregation.py"
)
_ex16 = _ilu.module_from_spec(_spec)  # type: ignore[arg-type]
assert _spec.loader is not None
_spec.loader.exec_module(_ex16)  # type: ignore[union-attr]

import geopandas as gpd  # noqa: E402

from sarenv.swarm.config import (  # noqa: E402
    DroneConfig,
    RobotDogConfig,
    SwarmConfig,
)
from sarenv.swarm.metrics import SwarmMetrics  # noqa: E402
from sarenv.swarm.simulator import SwarmSimulator  # noqa: E402


# ----------------------------------------------------------------------
# Run runner parametrizado (extiende run_one de ex16 con evap configurable)
# ----------------------------------------------------------------------

def run_one(
    *,
    label: str,
    item,
    num_drones: int,
    num_dogs: int,
    budget: float,
    max_steps: int,
    seed: int,
    num_victims: int,
    presence_weight: float,
    presence_evaporation: float,
) -> dict:
    print(
        f"  [run] {label}: w={presence_weight:.3f} evap={presence_evaporation:.3f} "
        f"drones={num_drones} dogs={num_dogs} budget={budget:.0f} "
        f"max_steps={max_steps} seed={seed}"
    )
    cfg = SwarmConfig(
        num_drones=num_drones,
        num_dogs=num_dogs,
        budget_per_agent=budget,
        max_steps=max_steps,
        presence_evaporation=presence_evaporation,
        drone_config=DroneConfig(
            altitude=80.0, fov_deg=45.0, presence_weight=presence_weight,
        ),
        dog_config=RobotDogConfig(
            sensor_range=20.0, presence_weight=presence_weight,
        ),
    )
    victims = _ex16.gen_victims(item, num_victims, seed)
    t0 = time.perf_counter()
    sim = SwarmSimulator.from_dataset_item(item, cfg, seed=seed)
    sim.run()
    elapsed = time.perf_counter() - t0
    metrics = SwarmMetrics(sim, victims=victims)
    rep = metrics.full_report()
    rep["scenario"] = label
    rep["presence_weight"] = presence_weight
    rep["presence_evaporation"] = presence_evaporation
    rep["num_drones"] = num_drones
    rep["num_dogs"] = num_dogs
    rep["budget"] = budget
    rep["seed"] = seed
    rep["elapsed_s"] = elapsed
    print(
        f"    prob_cov={rep.get('probability_coverage_ratio', 0):.4f} "
        f"cov={rep.get('coverage_ratio', 0):.4f} "
        f"eff={rep.get('efficiency_ratio', 0):.4f} "
        f"gini={rep.get('coverage_gini', 0):.4f} "
        f"cluster={rep.get('cluster_ratio', 0):.4f} "
        f"({elapsed:.1f}s)"
    )
    return rep


# ----------------------------------------------------------------------
# Barridos
# ----------------------------------------------------------------------

WEIGHT_GRID = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
EVAP_GRID = [0.01, 0.02, 0.05, 0.10]
SEED = 42


def sweep_weight() -> list[dict]:
    """Barre presence_weight en sarenv_s1 (drones puros, escenario rápido)."""
    print("\n=== Sweep WEIGHT en sarenv_s1 (5 drones, budget 100k) ===")
    item = _ex16.load_sarenv_scenario(1)
    if item is None:
        print("  [ABORT] no se pudo cargar sarenv_s1.")
        return []
    runs: list[dict] = []
    for w in WEIGHT_GRID:
        runs.append(
            run_one(
                label="sarenv_s1_5d_100k",
                item=item,
                num_drones=5,
                num_dogs=0,
                budget=100_000.0,
                max_steps=15_000,
                seed=SEED,
                num_victims=20,
                presence_weight=w,
                presence_evaporation=0.05,  # default iter3
            )
        )
    return runs


def sweep_evap(weight_star: float) -> list[dict]:
    """Barre presence_evaporation en maigmo con el weight ganador."""
    print(
        f"\n=== Sweep EVAP en maigmo (5 drones, budget 250k, w={weight_star}) ==="
    )
    item = _ex16.load_maigmo()
    if item is None:
        print("  [ABORT] no se pudo cargar maigmo.")
        return []
    runs: list[dict] = []
    # Incluir baseline w=0 para referencia
    runs.append(
        run_one(
            label="maigmo_5d_250k",
            item=item,
            num_drones=5,
            num_dogs=0,
            budget=250_000.0,
            max_steps=30_000,
            seed=SEED,
            num_victims=20,
            presence_weight=0.0,
            presence_evaporation=0.05,
        )
    )
    for ev in EVAP_GRID:
        runs.append(
            run_one(
                label="maigmo_5d_250k",
                item=item,
                num_drones=5,
                num_dogs=0,
                budget=250_000.0,
                max_steps=30_000,
                seed=SEED,
                num_victims=20,
                presence_weight=weight_star,
                presence_evaporation=ev,
            )
        )
    return runs


# ----------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------

KEY_COLS = [
    "scenario", "presence_weight", "presence_evaporation",
    "num_drones", "num_dogs", "budget", "seed",
    "probability_coverage_ratio", "coverage_ratio", "efficiency_ratio",
    "coverage_gini", "cluster_ratio", "mean_pair_distance_cells",
    "overlap_ratio", "effective_cells", "redundant_visits",
    "info_propagation_latency", "elapsed_s",
]


def write_csv(runs: list[dict], path: Path) -> None:
    if not runs:
        print(f"  [SKIP] sin runs, no se escribe {path}")
        return
    # Unión de claves de todos los runs preservando KEY_COLS al inicio
    all_keys = list(KEY_COLS)
    for r in runs:
        for k in r.keys():
            if k not in all_keys:
                all_keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            w.writerow(r)
    print(f"\n[OK] CSV escrito en {path}")


def print_summary(runs: list[dict], xkey: str) -> None:
    print(f"\n=== Resumen para docs/16 (xkey={xkey}) ===")
    print(
        f"| {xkey} | prob_cov | cov | eff | gini | cluster | meanD |"
    )
    print("|---|---|---|---|---|---|---|")
    for r in runs:
        print(
            f"| {r.get(xkey, 0):.3f} "
            f"| {r.get('probability_coverage_ratio', 0):.4f} "
            f"| {r.get('coverage_ratio', 0):.4f} "
            f"| {r.get('efficiency_ratio', 0):.4f} "
            f"| {r.get('coverage_gini', 0):.4f} "
            f"| {r.get('cluster_ratio', 0):.4f} "
            f"| {r.get('mean_pair_distance_cells', 0):.1f} |"
        )
    if runs:
        best = max(runs, key=lambda r: r.get("probability_coverage_ratio", 0))
        print(
            f"\n[BEST] {xkey}={best.get(xkey)} "
            f"prob_cov={best.get('probability_coverage_ratio', 0):.4f} "
            f"eff={best.get('efficiency_ratio', 0):.4f}"
        )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "weight"
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    if mode == "weight":
        runs = sweep_weight()
        write_csv(runs, out_dir / "iter3b_sweep_weight.csv")
        print_summary(runs, "presence_weight")
    elif mode == "evap":
        # weight_star puede pasarse como sys.argv[2]; default 0.05
        weight_star = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
        runs = sweep_evap(weight_star)
        write_csv(runs, out_dir / "iter3b_sweep_evap.csv")
        print_summary(runs, "presence_evaporation")
    else:
        print(f"[ERROR] modo desconocido: {mode}. Usa 'weight' o 'evap'.")
        sys.exit(2)


if __name__ == "__main__":
    main()
