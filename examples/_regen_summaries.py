"""Regenera summary/aggregate/tests desde iter4b_strat.csv (120 runs completos).
No importa el script principal para evitar cargar el dataset sarenv entero.
"""
import csv
import math
from pathlib import Path
from statistics import mean, stdev

try:
    from scipy.stats import wilcoxon  # type: ignore
except Exception:
    wilcoxon = None

ROOT = Path(__file__).resolve().parent.parent


# ── funciones copiadas de 20_iter4b_stratified.py ─────────────────────────────

def summarize(runs):
    groups = {}
    for r in runs:
        groups.setdefault((r["scenario"], r["do_kill"], r["presence_weight"]), []).append(r)
    metrics = ["probability_coverage_ratio", "coverage_ratio",
               "efficiency_ratio", "coverage_gini",
               "recovery_time", "recovery_coverage_at_end"]
    out = []
    for (scen, kill, w), items in groups.items():
        row = {"scenario": scen, "do_kill": kill, "presence_weight": w, "n": len(items)}
        for m in metrics:
            vals = [r.get(m) for r in items]
            vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
            row[f"{m}_n_finite"] = len(vals)
            row[f"{m}_mean"] = mean(vals) if vals else float("nan")
            row[f"{m}_std"] = stdev(vals) if len(vals) > 1 else 0.0
        out.append(row)
    return out


def aggregate_across_scenarios(runs):
    groups = {}
    for r in runs:
        groups.setdefault((r["do_kill"], r["presence_weight"]), []).append(r)
    metrics = ["probability_coverage_ratio", "efficiency_ratio",
               "recovery_time", "recovery_coverage_at_end"]
    out = []
    for (kill, w), items in groups.items():
        row = {"do_kill": kill, "presence_weight": w, "n": len(items)}
        for m in metrics:
            vals = [r.get(m) for r in items]
            vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
            row[f"{m}_n_finite"] = len(vals)
            row[f"{m}_mean"] = mean(vals) if vals else float("nan")
            row[f"{m}_std"] = stdev(vals) if len(vals) > 1 else 0.0
        out.append(row)
    return out


def paired_test_per_scenario(runs):
    out = []
    by_scen = {}
    for r in runs:
        if r["do_kill"]:
            by_scen.setdefault(r["scenario"], []).append(r)
    for scen, rs in by_scen.items():
        seeds = sorted({r["seed"] for r in rs})
        for metric in ["probability_coverage_ratio", "recovery_time", "recovery_coverage_at_end"]:
            a, b = [], []
            for s in seeds:
                ra = next((r for r in rs if r["seed"] == s and r["presence_weight"] == 0.0), None)
                rb = next((r for r in rs if r["seed"] == s and r["presence_weight"] == 0.01), None)
                if ra is None or rb is None:
                    continue
                va, vb = ra.get(metric), rb.get(metric)
                if metric == "recovery_time":
                    if isinstance(va, float) and math.isnan(va):
                        va = ra["max_steps"] - ra["kill_tick"]
                    if isinstance(vb, float) and math.isnan(vb):
                        vb = rb["max_steps"] - rb["kill_tick"]
                if va is None or vb is None:
                    continue
                if isinstance(va, float) and math.isnan(va): continue
                if isinstance(vb, float) and math.isnan(vb): continue
                a.append(float(va)); b.append(float(vb))
            if len(a) < 2:
                continue
            row = {"scenario": scen, "metric": metric, "n": len(a),
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


# ── carga del CSV ──────────────────────────────────────────────────────────────

all_runs = []
for row in csv.DictReader(open(ROOT / "results" / "iter4b_strat.csv")):
    row["do_kill"] = row["do_kill"].lower() in ("true", "1")
    row["presence_weight"] = float(row["presence_weight"])
    row["seed"] = int(row["seed"])
    for k in ("probability_coverage_ratio", "coverage_ratio", "efficiency_ratio",
              "coverage_gini", "recovery_time", "recovery_coverage_at_end",
              "max_steps", "kill_tick"):
        v = row.get(k, "")
        try:
            row[k] = float(v)
        except (ValueError, TypeError):
            row[k] = float("nan")
    all_runs.append(row)

print(f"Loaded {len(all_runs)} runs")

summary   = summarize(all_runs)
aggregate = aggregate_across_scenarios(all_runs)
tests     = paired_test_per_scenario(all_runs)


def write(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] {path}")


write(ROOT / "results" / "iter4b_strat_summary.csv",   summary)
write(ROOT / "results" / "iter4b_strat_aggregate.csv", aggregate)
write(ROOT / "results" / "iter4b_strat_tests.csv",     tests)

print()
print("=== Agregado 10 escenarios (mean +/- std) ===")
print("| kill | w | n | prob_cov | eff | recov_t | recov_cov |")
print("|---|---|---|---|---|---|---|")
for r in sorted(aggregate, key=lambda x: (x["do_kill"], x["presence_weight"])):
    rt = (f"{r['recovery_time_mean']:.0f}+/-{r['recovery_time_std']:.0f}"
          if not math.isnan(r["recovery_time_mean"]) else "n/a")
    rc = (f"{r['recovery_coverage_at_end_mean']:.2f}+/-{r['recovery_coverage_at_end_std']:.2f}"
          if not math.isnan(r["recovery_coverage_at_end_mean"]) else "n/a")
    print(
        f"| {r['do_kill']} | {r['presence_weight']:.2f} | {r['n']} "
        f"| {r['probability_coverage_ratio_mean']:.4f}+/-{r['probability_coverage_ratio_std']:.4f} "
        f"| {r['efficiency_ratio_mean']:.4f} | {rt} | {rc} |"
    )

print()
print("=== recovery_coverage_at_end por escenario (kill=True) ===")
print("| escenario | rc w=0 | rc w=0.01 | delta |")
print("|---|---|---|---|")
kill_rows = {(r["scenario"], r["presence_weight"]): r for r in summary if r["do_kill"]}
scens = sorted({r["scenario"] for r in all_runs})
for s in scens:
    r0  = kill_rows.get((s, 0.0))
    r01 = kill_rows.get((s, 0.01))
    if r0 and r01:
        v0  = r0["recovery_coverage_at_end_mean"]
        v01 = r01["recovery_coverage_at_end_mean"]
        print(f"| {s} | {v0:.2f} | {v01:.2f} | {v01-v0:+.2f} |")

print()
print("=== Wilcoxon (kill=True, w=0 vs w=0.01) ===")
print("| escenario | metrica | delta | p |")
print("|---|---|---|---|")
for t in tests:
    p = t.get("p_value")
    ps = f"{p:.3f}" if p is not None else "n/a"
    print(f"| {t['scenario']} | {t['metric']} | {t['delta']:+.4f} | {ps} |")
