"""Iter4.a — Tolerancia a fallos: ¿la feromona acelera la recuperación?

Predicción falsable (docs/16 §6.5 → §6.6):
    Si un agente cae a t=kill_tick, la feromona depositada en su zona
    deja de renovarse y se evapora con tasa `presence_evaporation=0.05`.
    A los ~1/0.05 = 20 ticks la zona vuelve a ser atractiva (presencia
    baja → score alto) y un agente vivo cercano la reincorpora.
    Sin feromona los demás agentes no tienen ninguna razón para acercarse;
    la zona huérfana queda sin cubrir hasta el azar.

Diseño:
    Escenario fijo: sarenv_s1, 5 drones, budget=100k, max_steps=15_000.
    Kill: drone_2 a t = 0.25 * max_steps = 3750.
    Configs (2×2):
        (kill=False, w=0.00)  control 1: baseline sin daño
        (kill=False, w=0.01)  control 2: feromona sin daño
        (kill=True,  w=0.00)  daño puro
        (kill=True,  w=0.01)  daño + feromona  ← hipótesis
    N=5 seeds.

Métrica de recuperación:
    `recovery_zone` = celdas visibles desde la posición del agente en t=kill
                       (su sensor footprint en el momento de morir).
    `recovery_time` = primer t >= kill_tick en que CUALQUIER agente vivo
                       entra en la zona, en su sensor footprint propio.
                       NaN si nunca ocurre dentro del horizonte.
    `recovery_coverage_at_end` = fracción de la zona huérfana que ha sido
                       observada por algún agente vivo al final de la sim.

Salidas:
    results/iter4a_fault_tolerance.csv          (raw por run)
    results/iter4a_fault_tolerance_summary.csv  (mean/std)
    results/iter4a_fault_tolerance_tests.csv    (Wilcoxon pareado)
"""
from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("ex16", ROOT / "examples" / "16_baseline_aggregation.py")
_ex16 = _ilu.module_from_spec(_spec)  # type: ignore[arg-type]
assert _spec.loader is not None
_spec.loader.exec_module(_ex16)  # type: ignore[union-attr]

from sarenv.swarm.config import DroneConfig, RobotDogConfig, SwarmConfig  # noqa: E402
from sarenv.swarm.metrics import SwarmMetrics  # noqa: E402
from sarenv.swarm.simulator import SwarmSimulator  # noqa: E402

try:
    from scipy.stats import wilcoxon  # type: ignore
except Exception:  # pragma: no cover
    wilcoxon = None


SEEDS_DEFAULT = [42, 1, 7, 100, 999, 13, 21, 77, 314, 2718]
KILL_AGENT_ID = "drone_2"
KILL_TICK = 1000  # fijo, antes de que se agote el presupuesto típico (~2500-3000 ticks)


def run_one(*, item, label, num_drones, budget, max_steps,
            seed, num_victims, presence_weight, do_kill: bool) -> dict:
    cfg = SwarmConfig(
        num_drones=num_drones,
        num_dogs=0,
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

    kill_tick = KILL_TICK if do_kill else -1
    recovery_zone: set[tuple[int, int]] = set()
    recovery_zone_visited: set[tuple[int, int]] = set()
    recovery_time: int | float = math.nan
    killed = False

    max_steps_actual = max_steps
    for t in range(max_steps_actual):
        # Kill antes del step para que el snapshot de zona sea pre-acción
        if do_kill and not killed and t == kill_tick:
            target = next((a for a in sim.agents if a.id == KILL_AGENT_ID and a.active), None)
            if target is not None:
                recovery_zone = set(target.get_visible_cells())
                ok = sim.kill_agent(KILL_AGENT_ID)
                killed = ok

        sim.step()

        if killed and recovery_zone and math.isnan(recovery_time):
            for a in sim.agents:
                if not a.active or a.id == KILL_AGENT_ID:
                    continue
                vis = a.get_visible_cells()
                hit = vis & recovery_zone
                if hit:
                    recovery_time = (t + 1) - kill_tick
                    recovery_zone_visited |= hit
                    break

        # Track ongoing coverage of orphan zone for the rest
        if killed and recovery_zone and not math.isnan(recovery_time):
            for a in sim.agents:
                if not a.active or a.id == KILL_AGENT_ID:
                    continue
                recovery_zone_visited |= (a.get_visible_cells() & recovery_zone)

        if not any(a.active for a in sim.agents):
            break

    elapsed = time.perf_counter() - t0
    rep = SwarmMetrics(sim, victims=victims).full_report()
    rep["scenario"] = label
    rep["presence_weight"] = presence_weight
    rep["seed"] = seed
    rep["do_kill"] = do_kill
    rep["kill_tick"] = kill_tick
    rep["recovery_zone_size"] = len(recovery_zone)
    rep["recovery_time"] = recovery_time
    rep["recovery_coverage_at_end"] = (
        len(recovery_zone_visited) / len(recovery_zone)
        if recovery_zone else float("nan")
    )
    rep["elapsed_s"] = elapsed
    rt_str = f"{recovery_time:.0f}" if not (isinstance(recovery_time, float) and math.isnan(recovery_time)) else "NaN"
    rcov = rep["recovery_coverage_at_end"]
    rcov_str = f"{rcov:.2f}" if not (isinstance(rcov, float) and math.isnan(rcov)) else "NaN"
    print(
        f"  [{label} seed={seed:>3} w={presence_weight:.2f} kill={do_kill}] "
        f"prob_cov={rep.get('probability_coverage_ratio', 0):.4f} "
        f"recov_t={rt_str:>5} recov_cov={rcov_str} "
        f"({elapsed:.1f}s)"
    )
    return rep


def summarize(runs: list[dict]) -> list[dict]:
    """Agrupa por (do_kill, presence_weight)."""
    groups: dict[tuple[bool, float], list[dict]] = {}
    for r in runs:
        key = (r["do_kill"], r["presence_weight"])
        groups.setdefault(key, []).append(r)

    metrics = [
        "probability_coverage_ratio",
        "coverage_ratio",
        "efficiency_ratio",
        "coverage_gini",
        "recovery_time",
        "recovery_coverage_at_end",
    ]
    summary: list[dict] = []
    for (kill, w), items in groups.items():
        row: dict = {"do_kill": kill, "presence_weight": w, "n": len(items)}
        for m in metrics:
            vals = [r.get(m) for r in items]
            vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
            row[f"{m}_n_finite"] = len(vals)
            row[f"{m}_mean"] = mean(vals) if vals else float("nan")
            row[f"{m}_std"] = stdev(vals) if len(vals) > 1 else 0.0
        summary.append(row)
    return summary


def paired_test(runs: list[dict]) -> list[dict]:
    """Compara kill_w0 vs kill_w001 (efecto de la feromona BAJO daño).

    Pareamos por seed dentro del subconjunto do_kill=True.
    """
    out: list[dict] = []
    kill_runs = [r for r in runs if r["do_kill"]]
    seeds = sorted({r["seed"] for r in kill_runs})

    for m in ["probability_coverage_ratio", "efficiency_ratio",
              "recovery_time", "recovery_coverage_at_end"]:
        a, b = [], []  # a = w=0, b = w=0.01
        for s in seeds:
            ra = next((r for r in kill_runs if r["seed"] == s and r["presence_weight"] == 0.0), None)
            rb = next((r for r in kill_runs if r["seed"] == s and r["presence_weight"] == 0.01), None)
            if ra is None or rb is None:
                continue
            va, vb = ra.get(m), rb.get(m)
            # Si recovery_time es NaN, lo tratamos como max_steps (penalización máxima)
            if m == "recovery_time":
                if isinstance(va, float) and math.isnan(va):
                    va = ra["kill_tick"] and (15_000 - ra["kill_tick"])
                if isinstance(vb, float) and math.isnan(vb):
                    vb = rb["kill_tick"] and (15_000 - rb["kill_tick"])
            if va is None or vb is None:
                continue
            a.append(float(va))
            b.append(float(vb))
        if len(a) < 2:
            continue
        row = {
            "metric": m, "n": len(a),
            "mean_w0": mean(a), "mean_w001": mean(b),
            "delta": mean(b) - mean(a),
            "delta_pct": (mean(b) - mean(a)) / mean(a) * 100.0 if mean(a) != 0 else 0.0,
        }
        if wilcoxon is not None and len(a) >= 3 and any(x != y for x, y in zip(a, b)):
            try:
                stat, p = wilcoxon(b, a)
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


def print_tables(summary: list[dict], tests: list[dict]) -> None:
    print("\n=== Resumen mean ± std por configuración (sarenv_s1, 5 drones) ===")
    print("| kill | w | n | prob_cov | eff | recov_t | recov_cov |")
    print("|---|---|---|---|---|---|---|")
    for r in summary:
        rt_m, rt_s = r["recovery_time_mean"], r["recovery_time_std"]
        rt_str = f"{rt_m:.0f}±{rt_s:.0f}" if not math.isnan(rt_m) else "n/a"
        rc_m, rc_s = r["recovery_coverage_at_end_mean"], r["recovery_coverage_at_end_std"]
        rc_str = f"{rc_m:.2f}±{rc_s:.2f}" if not math.isnan(rc_m) else "n/a"
        print(
            f"| {r['do_kill']} | {r['presence_weight']:.2f} | {r['n']} "
            f"| {r['probability_coverage_ratio_mean']:.4f}±{r['probability_coverage_ratio_std']:.4f} "
            f"| {r['efficiency_ratio_mean']:.4f}±{r['efficiency_ratio_std']:.4f} "
            f"| {rt_str} | {rc_str} |"
        )

    print("\n=== Wilcoxon pareado (BAJO daño: w=0.01 vs w=0) ===")
    print("| métrica | mean(w=0) | mean(w=0.01) | Δ | Δ% | p-value |")
    print("|---|---|---|---|---|---|")
    for r in tests:
        p = r.get("p_value")
        p_str = f"{p:.4f}" if p is not None else "n/a"
        print(
            f"| {r['metric']} | {r['mean_w0']:.4f} | {r['mean_w001']:.4f} "
            f"| {r['delta']:+.4f} | {r['delta_pct']:+.1f}% | {p_str} |"
        )


def main() -> None:
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    seeds = SEEDS_DEFAULT[:n_seeds]
    print(f"[cfg] seeds={seeds}  KILL_AGENT={KILL_AGENT_ID}  KILL_TICK={KILL_TICK}")

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    item_s1 = _ex16.load_sarenv_scenario(1)
    if item_s1 is None:
        print("[ERR] no se pudo cargar sarenv scenario 1")
        sys.exit(2)

    label = "sarenv_s1_5d_100k"
    runs: list[dict] = []
    print(f"\n=== {label} (kill={KILL_AGENT_ID} @ t={KILL_TICK}/15000) ===")
    for seed in seeds:
        for do_kill in (False, True):
            for w in (0.00, 0.01):
                runs.append(
                    run_one(
                        item=item_s1, label=label,
                        num_drones=5, budget=100_000, max_steps=15_000,
                        seed=seed, num_victims=250,
                        presence_weight=w, do_kill=do_kill,
                    )
                )

    write_csv(runs, out_dir / "iter4a_fault_tolerance.csv")
    summary = summarize(runs)
    write_csv(summary, out_dir / "iter4a_fault_tolerance_summary.csv")
    tests = paired_test(runs)
    write_csv(tests, out_dir / "iter4a_fault_tolerance_tests.csv")
    print_tables(summary, tests)


if __name__ == "__main__":
    main()
