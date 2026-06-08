"""Iter4.b — Tolerancia a fallos en 10 escenarios estratificados.

Generaliza iter4.a más allá de `sarenv_s1` para ver si el trade-off
(más cobertura promedio pero peor recuperación tras kill) se confirma
universalmente o depende del escenario.

Estratificación de los 60 escenarios sarenv:
    - flat temperate S  (660²,  ids 1-15)  → 3 elegidos: 1, 5, 10
    - flat dry M        (873²,  ids 31-45) → 3 elegidos: 31, 35, 40
    - mountainous temp. L (1286², ids 16-30) → 2: 16, 25
    - mountainous dry L (1286², ids 46-60) → 2: 46, 55

Diseño: 10 escenarios × 5 seeds × 4 configs (kill_F/T × w_0/0.01) = 200 runs.
Budget escalado por área para que las trayectorias sean comparables:
    budget = 100_000 · (max_dim / 660)
    max_steps = 15_000 · (max_dim / 660)
Kill: drone_2 a t = 0.20 · max_steps (dentro del horizonte útil).

Salidas:
    results/iter4b_strat.csv          (raw por run)
    results/iter4b_strat_summary.csv  (mean/std por (escenario, do_kill, w))
    results/iter4b_strat_aggregate.csv (mean/std agregado por (do_kill, w) sobre todos los escenarios)
    results/iter4b_strat_tests.csv    (Wilcoxon pareado por escenario, BAJO daño)
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
# kill_tick absoluto: temprano para garantizar que drone_2 esté activo
# (con KILL_FRACTION del max_steps caía tras agotar budget en mapas pequeños).
# 500 ticks = mismo orden que iter4.a (1000) pero margen extra.
KILL_TICK_ABS = 500

# Estratificación: (id, etiqueta_corta, env_type, climate)
STRATA = [
    (1,  "flatT_S_01",  "flat",        "temperate"),
    (5,  "flatT_S_05",  "flat",        "temperate"),
    (10, "flatT_S_10",  "flat",        "temperate"),
    (31, "flatD_M_31",  "flat",        "dry"),
    (35, "flatD_M_35",  "flat",        "dry"),
    (40, "flatD_M_40",  "flat",        "dry"),
    (16, "mntT_L_16",   "mountainous", "temperate"),
    (25, "mntT_L_25",   "mountainous", "temperate"),
    (46, "mntD_L_46",   "mountainous", "dry"),
    (55, "mntD_L_55",   "mountainous", "dry"),
]


def scale_for(item) -> tuple[int, int, int]:
    """Devuelve (budget, max_steps, kill_tick) escalados al tamaño del mapa."""
    h = item.heatmap
    max_dim = max(h.shape)
    factor = max_dim / 660.0
    budget = int(100_000 * factor)
    max_steps = int(15_000 * factor)
    kill_tick = KILL_TICK_ABS  # absoluto para garantizar agente vivo
    return budget, max_steps, kill_tick


def run_one(*, item, label, env_type, climate, num_drones,
            budget, max_steps, kill_tick,
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

    kill_at = kill_tick if do_kill else -1
    recovery_zone: set[tuple[int, int]] = set()
    recovery_zone_visited: set[tuple[int, int]] = set()
    recovery_time: int | float = math.nan
    killed = False

    for t in range(max_steps):
        if do_kill and not killed and t == kill_at:
            target = next((a for a in sim.agents if a.id == KILL_AGENT_ID and a.active), None)
            if target is not None:
                recovery_zone = set(target.get_visible_cells())
                killed = sim.kill_agent(KILL_AGENT_ID)

        sim.step()

        if killed and recovery_zone:
            for a in sim.agents:
                if not a.active or a.id == KILL_AGENT_ID:
                    continue
                vis = a.get_visible_cells()
                hit = vis & recovery_zone
                if hit:
                    if math.isnan(recovery_time):
                        recovery_time = (t + 1) - kill_at
                    recovery_zone_visited |= hit

        if not any(a.active for a in sim.agents):
            break

    elapsed = time.perf_counter() - t0
    if do_kill and not killed:
        print(f"  [WARN] do_kill=True pero {KILL_AGENT_ID} ya inactivo en t={kill_at} (kill no aplicado)")
    rep = SwarmMetrics(sim, victims=victims).full_report()
    rep["scenario"] = label
    rep["env_type"] = env_type
    rep["climate"] = climate
    rep["presence_weight"] = presence_weight
    rep["seed"] = seed
    rep["do_kill"] = do_kill
    rep["kill_tick"] = kill_at
    rep["max_steps"] = max_steps
    rep["budget"] = budget
    rep["recovery_zone_size"] = len(recovery_zone)
    rep["kill_applied"] = bool(killed)
    rep["recovery_time"] = recovery_time
    rep["recovery_coverage_at_end"] = (
        len(recovery_zone_visited) / len(recovery_zone)
        if recovery_zone else float("nan")
    )
    rep["elapsed_s"] = elapsed
    rt_str = (f"{recovery_time:.0f}"
              if not (isinstance(recovery_time, float) and math.isnan(recovery_time))
              else "NaN")
    rcov = rep["recovery_coverage_at_end"]
    rcov_str = (f"{rcov:.2f}"
                if not (isinstance(rcov, float) and math.isnan(rcov))
                else "NaN")
    print(
        f"  [{label} seed={seed:>3} w={presence_weight:.2f} kill={int(do_kill)}] "
        f"prob={rep.get('probability_coverage_ratio', 0):.3f} "
        f"rt={rt_str:>5} rc={rcov_str} ({elapsed:.1f}s)"
    )
    return rep


def summarize(runs: list[dict]) -> list[dict]:
    """Agrupa por (scenario, do_kill, presence_weight)."""
    groups: dict[tuple[str, bool, float], list[dict]] = {}
    for r in runs:
        key = (r["scenario"], r["do_kill"], r["presence_weight"])
        groups.setdefault(key, []).append(r)

    metrics = ["probability_coverage_ratio", "coverage_ratio",
               "efficiency_ratio", "coverage_gini",
               "recovery_time", "recovery_coverage_at_end"]
    summary: list[dict] = []
    for (scen, kill, w), items in groups.items():
        row: dict = {"scenario": scen, "do_kill": kill, "presence_weight": w, "n": len(items)}
        for m in metrics:
            vals = [r.get(m) for r in items]
            vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
            row[f"{m}_n_finite"] = len(vals)
            row[f"{m}_mean"] = mean(vals) if vals else float("nan")
            row[f"{m}_std"] = stdev(vals) if len(vals) > 1 else 0.0
        summary.append(row)
    return summary


def aggregate_across_scenarios(runs: list[dict]) -> list[dict]:
    """Agrega TODOS los runs por (do_kill, presence_weight) — vista global."""
    groups: dict[tuple[bool, float], list[dict]] = {}
    for r in runs:
        groups.setdefault((r["do_kill"], r["presence_weight"]), []).append(r)
    metrics = ["probability_coverage_ratio", "efficiency_ratio",
               "recovery_time", "recovery_coverage_at_end"]
    out: list[dict] = []
    for (kill, w), items in groups.items():
        row: dict = {"do_kill": kill, "presence_weight": w, "n": len(items)}
        for m in metrics:
            vals = [r.get(m) for r in items]
            vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
            row[f"{m}_n_finite"] = len(vals)
            row[f"{m}_mean"] = mean(vals) if vals else float("nan")
            row[f"{m}_std"] = stdev(vals) if len(vals) > 1 else 0.0
        out.append(row)
    return out


def paired_test_per_scenario(runs: list[dict]) -> list[dict]:
    """Wilcoxon pareado (w=0 vs w=0.01) BAJO daño, por escenario."""
    out: list[dict] = []
    by_scen: dict[str, list[dict]] = {}
    for r in runs:
        if r["do_kill"]:
            by_scen.setdefault(r["scenario"], []).append(r)

    for scen, rs in by_scen.items():
        seeds = sorted({r["seed"] for r in rs})
        for m in ["probability_coverage_ratio", "recovery_time",
                  "recovery_coverage_at_end"]:
            a, b = [], []
            for s in seeds:
                ra = next((r for r in rs if r["seed"] == s and r["presence_weight"] == 0.0), None)
                rb = next((r for r in rs if r["seed"] == s and r["presence_weight"] == 0.01), None)
                if ra is None or rb is None:
                    continue
                va, vb = ra.get(m), rb.get(m)
                if m == "recovery_time":
                    if isinstance(va, float) and math.isnan(va):
                        va = ra["max_steps"] - ra["kill_tick"]
                    if isinstance(vb, float) and math.isnan(vb):
                        vb = rb["max_steps"] - rb["kill_tick"]
                if va is None or vb is None:
                    continue
                if isinstance(va, float) and math.isnan(va): continue
                if isinstance(vb, float) and math.isnan(vb): continue
                a.append(float(va)); b.append(float(vb))
            if len(a) < 2: continue
            row = {"scenario": scen, "metric": m, "n": len(a),
                   "mean_w0": mean(a), "mean_w001": mean(b),
                   "delta": mean(b) - mean(a),
                   "delta_pct": (mean(b) - mean(a)) / mean(a) * 100.0 if mean(a) != 0 else 0.0}
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
    if not rows: return
    keys: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys: keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"[OK] CSV escrito en {path}")


def print_aggregate(agg: list[dict]) -> None:
    print("\n=== Agregado a través de los 10 escenarios (mean ± std) ===")
    print("| kill | w | n | prob_cov | eff | recov_t | recov_cov |")
    print("|---|---|---|---|---|---|---|")
    for r in agg:
        rt_m = r["recovery_time_mean"]; rt_s = r["recovery_time_std"]
        rt_str = f"{rt_m:.0f}±{rt_s:.0f}" if not math.isnan(rt_m) else "n/a"
        rc_m = r["recovery_coverage_at_end_mean"]; rc_s = r["recovery_coverage_at_end_std"]
        rc_str = f"{rc_m:.2f}±{rc_s:.2f}" if not math.isnan(rc_m) else "n/a"
        print(
            f"| {r['do_kill']} | {r['presence_weight']:.2f} | {r['n']} "
            f"| {r['probability_coverage_ratio_mean']:.4f}±{r['probability_coverage_ratio_std']:.4f} "
            f"| {r['efficiency_ratio_mean']:.4f}±{r['efficiency_ratio_std']:.4f} "
            f"| {rt_str} | {rc_str} |"
        )


def print_per_scenario(summary: list[dict]) -> None:
    print("\n=== Por escenario, recovery_coverage_at_end (kill=True) ===")
    print("| escenario | recov_cov w=0 | recov_cov w=0.01 | Δ |")
    print("|---|---|---|---|")
    by = {}
    for r in summary:
        if not r["do_kill"]: continue
        by.setdefault(r["scenario"], {})[r["presence_weight"]] = r
    for scen in sorted(by):
        r0 = by[scen].get(0.0); r1 = by[scen].get(0.01)
        if r0 is None or r1 is None: continue
        a = r0["recovery_coverage_at_end_mean"]; b = r1["recovery_coverage_at_end_mean"]
        d = b - a if not (math.isnan(a) or math.isnan(b)) else float("nan")
        print(f"| {scen} | {a:.2f} | {b:.2f} | {d:+.2f} |")


def main() -> None:
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    seeds = SEEDS_DEFAULT[:n_seeds]
    print(f"[cfg] seeds={seeds}  KILL_AGENT={KILL_AGENT_ID}  KILL_TICK_ABS={KILL_TICK_ABS}")
    print(f"[cfg] STRATA: {[s[1] for s in STRATA]}")

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    runs: list[dict] = []
    t_global = time.perf_counter()

    for scen_id, label, env_type, climate in STRATA:
        item = _ex16.load_sarenv_scenario(scen_id)
        if item is None:
            print(f"[skip] no se pudo cargar escenario {scen_id}")
            continue
        budget, max_steps, kill_tick = scale_for(item)
        h_shape = item.heatmap.shape
        print(f"\n=== {label} (id={scen_id}, shape={h_shape}, "
              f"budget={budget}, max_steps={max_steps}, kill@{kill_tick}) ===")
        for seed in seeds:
            for do_kill in (False, True):
                for w in (0.00, 0.01):
                    runs.append(
                        run_one(
                            item=item, label=label,
                            env_type=env_type, climate=climate,
                            num_drones=5,
                            budget=budget, max_steps=max_steps,
                            kill_tick=kill_tick,
                            seed=seed, num_victims=250,
                            presence_weight=w, do_kill=do_kill,
                        )
                    )
        # write progressive snapshot for safety (long run)
        write_csv(runs, out_dir / "iter4b_strat.csv")

    elapsed_global = time.perf_counter() - t_global
    print(f"\n[wall] total={elapsed_global/60:.1f} min  total_runs={len(runs)}")

    write_csv(runs, out_dir / "iter4b_strat.csv")
    summary = summarize(runs)
    write_csv(summary, out_dir / "iter4b_strat_summary.csv")
    agg = aggregate_across_scenarios(runs)
    write_csv(agg, out_dir / "iter4b_strat_aggregate.csv")
    tests = paired_test_per_scenario(runs)
    write_csv(tests, out_dir / "iter4b_strat_tests.csv")

    print_aggregate(agg)
    print_per_scenario(summary)


if __name__ == "__main__":
    main()
