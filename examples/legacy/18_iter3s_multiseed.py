"""Iter3.s — Validación estadística multi-seed de la feromona de presencia.

Compara baseline (`presence_weight=0`) vs iter3.b (`presence_weight=0.01`)
en sarenv_s1 y maigmo a lo largo de varias seeds para verificar que la
mejora medida en iter3.b NO es ruido.

Ejecuta:
    python examples/18_iter3s_multiseed.py [n_seeds]

Por defecto n_seeds=5 → seeds = [42, 1, 7, 100, 999].

Salida:
    results/iter3s_multiseed.csv          (todos los runs)
    results/iter3s_multiseed_summary.csv  (media/std/p-value por config)
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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

try:
    from scipy.stats import wilcoxon  # type: ignore
except Exception:  # pragma: no cover
    wilcoxon = None


SEEDS_DEFAULT = [42, 1, 7, 100, 999, 13, 21, 77, 314, 2718]
KEY_METRICS = [
    "probability_coverage_ratio",
    "coverage_ratio",
    "efficiency_ratio",
    "coverage_gini",
    "cluster_ratio",
    "mean_pair_distance_cells",
]


def run_one(*, item, label, num_drones, num_dogs, budget, max_steps,
            seed, num_victims, presence_weight) -> dict:
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
    victims = _ex16.gen_victims(item, num_victims, seed)
    t0 = time.perf_counter()
    sim = SwarmSimulator.from_dataset_item(item, cfg, seed=seed)
    sim.run()
    elapsed = time.perf_counter() - t0
    rep = SwarmMetrics(sim, victims=victims).full_report()
    rep["scenario"] = label
    rep["presence_weight"] = presence_weight
    rep["seed"] = seed
    rep["elapsed_s"] = elapsed
    print(
        f"  [{label} seed={seed:>3} w={presence_weight:.2f}] "
        f"prob_cov={rep.get('probability_coverage_ratio', 0):.4f} "
        f"eff={rep.get('efficiency_ratio', 0):.4f} "
        f"cluster={rep.get('cluster_ratio', 0):.4f} "
        f"({elapsed:.1f}s)"
    )
    return rep


def run_scenario(*, label, item, num_drones, num_dogs, budget, max_steps,
                 num_victims, seeds) -> list[dict]:
    print(f"\n=== {label} ({len(seeds)} seeds × 2 configs) ===")
    runs: list[dict] = []
    for seed in seeds:
        for w in [0.00, 0.01]:
            runs.append(
                run_one(
                    item=item, label=label,
                    num_drones=num_drones, num_dogs=num_dogs,
                    budget=budget, max_steps=max_steps,
                    seed=seed, num_victims=num_victims,
                    presence_weight=w,
                )
            )
    return runs


def summarize(runs: list[dict]) -> list[dict]:
    """Agrupa por (scenario, presence_weight) y devuelve mean/std/n por métrica."""
    groups: dict[tuple[str, float], list[dict]] = {}
    for r in runs:
        key = (r["scenario"], r["presence_weight"])
        groups.setdefault(key, []).append(r)

    summary: list[dict] = []
    for (scenario, w), items in groups.items():
        row = {"scenario": scenario, "presence_weight": w, "n": len(items)}
        for m in KEY_METRICS:
            vals = [r.get(m, 0.0) for r in items]
            row[f"{m}_mean"] = mean(vals)
            row[f"{m}_std"] = stdev(vals) if len(vals) > 1 else 0.0
        summary.append(row)
    return summary


def paired_test(runs: list[dict]) -> list[dict]:
    """Wilcoxon paired test entre w=0 y w=0.01 por seed para cada escenario.

    Devuelve por escenario y métrica el p-value y la magnitud del efecto.
    """
    out: list[dict] = []
    by_scen: dict[str, list[dict]] = {}
    for r in runs:
        by_scen.setdefault(r["scenario"], []).append(r)

    for scen, rs in by_scen.items():
        # Pareamos por seed
        seeds = sorted({r["seed"] for r in rs})
        for m in KEY_METRICS:
            base = []
            iter3b = []
            for s in seeds:
                rb = next((r for r in rs if r["seed"] == s and r["presence_weight"] == 0.0), None)
                ri = next((r for r in rs if r["seed"] == s and r["presence_weight"] == 0.01), None)
                if rb is None or ri is None:
                    continue
                base.append(rb.get(m, 0.0))
                iter3b.append(ri.get(m, 0.0))
            if len(base) < 2:
                continue
            row = {"scenario": scen, "metric": m,
                   "n": len(base),
                   "baseline_mean": mean(base),
                   "iter3b_mean": mean(iter3b),
                   "delta_mean": mean(iter3b) - mean(base),
                   "delta_pct": ((mean(iter3b) - mean(base)) / mean(base) * 100.0)
                   if mean(base) != 0 else 0.0}
            if wilcoxon is not None and len(base) >= 3 and any(a != b for a, b in zip(base, iter3b)):
                try:
                    stat, p = wilcoxon(iter3b, base)
                    row["wilcoxon_stat"] = float(stat)
                    row["p_value"] = float(p)
                except Exception as exc:
                    row["p_value"] = None
                    row["wilcoxon_err"] = str(exc)
            else:
                row["p_value"] = None
            out.append(row)
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[OK] CSV escrito en {path}")


def print_summary_table(summary: list[dict], tests: list[dict]) -> None:
    print("\n=== Resumen mean ± std por configuración ===")
    print("| escenario | w | n | prob_cov | eff | cluster |")
    print("|---|---|---|---|---|---|")
    for r in summary:
        print(
            f"| {r['scenario']} | {r['presence_weight']:.2f} | {r['n']} "
            f"| {r['probability_coverage_ratio_mean']:.4f}±{r['probability_coverage_ratio_std']:.4f} "
            f"| {r['efficiency_ratio_mean']:.4f}±{r['efficiency_ratio_std']:.4f} "
            f"| {r['cluster_ratio_mean']:.4f}±{r['cluster_ratio_std']:.4f} |"
        )

    print("\n=== Test pareado Wilcoxon (w=0.01 vs w=0) ===")
    print("| escenario | métrica | Δ | Δ% | p-value |")
    print("|---|---|---|---|---|")
    for r in tests:
        if r["metric"] not in ("probability_coverage_ratio", "efficiency_ratio",
                               "cluster_ratio", "coverage_gini"):
            continue
        p = r.get("p_value")
        p_str = f"{p:.4f}" if p is not None else "n/a"
        print(
            f"| {r['scenario']} | {r['metric']} "
            f"| {r['delta_mean']:+.4f} | {r['delta_pct']:+.1f}% | {p_str} |"
        )


def main() -> None:
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    seeds = SEEDS_DEFAULT[:n_seeds]
    print(f"[cfg] seeds={seeds}")

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    runs: list[dict] = []

    item_s1 = _ex16.load_sarenv_scenario(1)
    if item_s1 is not None:
        runs += run_scenario(
            label="sarenv_s1_5d_100k", item=item_s1,
            num_drones=5, num_dogs=0,
            budget=100_000.0, max_steps=15_000,
            num_victims=20, seeds=seeds,
        )

    item_mg = _ex16.load_maigmo()
    if item_mg is not None:
        runs += run_scenario(
            label="maigmo_5d_250k", item=item_mg,
            num_drones=5, num_dogs=0,
            budget=250_000.0, max_steps=30_000,
            num_victims=20, seeds=seeds,
        )

    if not runs:
        print("Sin runs ejecutados.")
        return

    write_csv(runs, out_dir / "iter3s_multiseed.csv")
    summary = summarize(runs)
    tests = paired_test(runs)
    write_csv(summary, out_dir / "iter3s_multiseed_summary.csv")
    write_csv(tests, out_dir / "iter3s_multiseed_tests.csv")
    print_summary_table(summary, tests)


if __name__ == "__main__":
    main()
