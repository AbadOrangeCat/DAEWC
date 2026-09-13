"""Rebuild tables, paired uncertainty, feasibility, and Pareto data from run files."""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import t
from .protocol import feasible
from .run import write_json


NAMES = {"head": "Head-only", "full": "Full fine-tuning", "full_ewc": "Full fine-tuning + EWC",
         "adapter": "Adapter-only", "lora": "LoRA", "lwf": "LwF", "daewc": "DAEWC",
         "no_gate": "DAEWC without gate", "no_ewc": "DAEWC without EWC",
         "no_proximity": "DAEWC without proximity", "no_regularizers": "DAEWC without either penalty"}


def interval(values):
    x = np.asarray(values, dtype=float)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else None,
            "ci95_halfwidth": float(t.ppf(.975, len(x)-1) * x.std(ddof=1) / len(x)**.5) if len(x) > 1 else None,
            "n": len(x)}


def csv_write(path, rows):
    if not rows: return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def summarize(run_root, seq_root, out):
    out.mkdir(parents=True, exist_ok=True)
    runs = [json.loads(p.read_text()) for p in sorted((run_root / "runs").glob("*.json"))]
    grouped = defaultdict(list)
    raw = []
    for r in runs:
        grouped[(r["domain"], r["shots_per_class"], r["method"])].append(r)
        raw.append({"run_id": r["run_id"], "domain": r["domain"], "shots": r["shots_per_class"], "seed": r["seed"], "method": r["method"],
                    "target_f1": r["target"]["all"]["macro_f1"], "matched_f1": r["target"]["matched"]["macro_f1"] if r["target"]["matched"] else None,
                    "source_before_f1": r["source_before"]["all"]["macro_f1"], "source_after_f1": r["source_after"]["all"]["macro_f1"],
                    "delta_source_pp": r["delta_source_pp"], "trainable_parameters": r["training"]["trainable_parameters"],
                    "adapt_seconds": r["training"]["seconds"], "checkpoint": r["checkpoint"]})
    summary, feasibility, pareto = [], [], []
    for (domain, shots, method), group in sorted(grouped.items()):
        target = interval([r["target"]["all"]["macro_f1"] for r in group])
        change = interval([r["delta_source_pp"] for r in group])
        matched = interval([r["target"]["matched"]["macro_f1"] for r in group if r["target"]["matched"]])
        row = {"domain": domain, "shots": shots, "method": method, "method_name": NAMES[method],
               **{f"target_{k}": v for k, v in target.items()},
               **{f"source_change_{k}": v for k, v in change.items()},
               **{f"matched_{k}": v for k, v in matched.items()},
               "trainable_parameters": group[0]["training"]["trainable_parameters"],
               "adapt_seconds_mean": float(np.mean([r["training"]["seconds"] for r in group]))}
        summary.append(row)
        pareto.append({"domain": domain, "shots": shots, "method": method,
                       "target_mean": target["mean"], "source_change_mean_pp": change["mean"]})
        for tolerance in [.5, 1., 2., 5.]:
            for absolute in [False, True]:
                mask = [feasible(r["delta_source_pp"], tolerance, absolute) for r in group]
                feasibility.append({"domain": domain, "shots": shots, "method": method,
                                    "tolerance_pp": tolerance, "rule": "absolute_band" if absolute else "loss_only",
                                    "feasible_runs": sum(mask), "total_runs": len(mask),
                                    "all_runs_feasible": all(mask), "target_mean_all_runs": target["mean"]})
    for row in pareto:
        others = [x for x in pareto if x["domain"] == row["domain"] and x["shots"] == row["shots"]]
        row["non_dominated"] = not any(x["target_mean"] >= row["target_mean"] and x["source_change_mean_pp"] >= row["source_change_mean_pp"] and
                                         (x["target_mean"] > row["target_mean"] or x["source_change_mean_pp"] > row["source_change_mean_pp"]) for x in others)
    paired = []
    for (domain, shots, method), group in sorted(grouped.items()):
        if method == "daewc": continue
        ours = {r["seed"]: r for r in grouped.get((domain, shots, "daewc"), [])}
        shared = [r for r in group if r["seed"] in ours]
        if shared:
            paired.append({"domain": domain, "shots": shots, "comparison": f"DAEWC minus {NAMES[method]}",
                           **interval([ours[r["seed"]]["target"]["all"]["macro_f1"] - r["target"]["all"]["macro_f1"] for r in shared])})
    for name, table in [("runs.csv", raw), ("summary.csv", summary), ("retention_sensitivity.csv", feasibility),
                        ("pareto.csv", pareto), ("paired_differences.csv", paired)]:
        csv_write(out / name, table)
    sequences = [json.loads(p.read_text()) for p in sorted((seq_root / "runs").glob("*.json"))] if seq_root else []
    seq_groups = defaultdict(list)
    for r in sequences: seq_groups[r["method"]].append(r)
    seq_summary = []
    for method, group in seq_groups.items():
        # Average over the six orders within each seed, then compute seed uncertainty.
        by_seed = defaultdict(list)
        for r in group: by_seed[r["seed"]].append(r)
        values = {field: [float(np.mean([r[field] for r in rs])) for rs in by_seed.values()]
                  for field in ["bwt_older_targets_pp", "delta_source_pp", "final_target_macro_f1"]}
        seq_summary.append({"method": method, "sequences": len(group),
                            **{f"{field}_{k}": v for field, vs in values.items() for k, v in interval(vs).items()}})
    csv_write(out / "sequential_summary.csv", seq_summary)
    write_json(out / "summary.json", {"run_count": len(runs), "groups": summary,
                                       "sequences": seq_summary, "sequence_count": len(sequences),
                                       "uncertainty": "Student t interval across paired seeds, conditional on fixed test set; not test-sampling uncertainty"})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--sequential", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    summarize(args.runs, args.sequential, args.out)
