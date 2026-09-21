"""Verify the six final-model groups using saved predictions, without model weights."""
import argparse
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

project = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project))
from daewc.data import sha256
from daewc.run import code_hash
from daewc.verify import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=project / "revision_artifacts")
    parser.add_argument("--out", type=Path, default=project / "verification_output/saved_results.json")
    args = parser.parse_args()
    root = args.artifacts
    load = lambda path: json.loads(path.read_text())
    counts = {"local": 288, "budget_cv": 126, "random": 45,
              "bert_base": 63, "low_footprint": 108, "low_footprint_cv": 36}
    sources = {"budget_cv": "local", "low_footprint_cv": "low_footprint"}
    training_hash = code_hash()
    data_hash = sha256(root / "data/records.jsonl")
    paired = {}
    reports = {}
    for group, count in counts.items():
        folder = root / group
        plan = load(folder / "frozen_plan.json")
        if plan.get("core_code_hash", plan.get("code_sha256")) != training_hash:
            raise ValueError(f"{group}: frozen training code differs")
        if plan.get("data_hash", plan.get("data_sha256")) != data_hash:
            raise ValueError(f"{group}: frozen prepared data differ")
        if "base_configuration" in plan:
            cfg = plan["base_configuration"]
            wanted = set(itertools.product(cfg["domains"], plan["shots"], cfg["seeds"], plan["methods"]))
            if plan["cv_code_hash"] != sha256(project / "daewc/budget_cv.py"):
                raise ValueError(f"{group}: frozen selection code differs")
        else:
            cfg = plan["configuration"]
            wanted = set(itertools.product(cfg["domains"], cfg["shots"], cfg["seeds"], cfg["methods"]))
            wanted.update(itertools.product(cfg["domains"], cfg.get("ablation_shots", []),
                                            cfg["seeds"], cfg.get("ablation_methods", [])))
        runs = [load(path) for path in (folder / "runs").glob("*.json")]
        observed = {(r["domain"], r["shots_per_class"], r["seed"], r["method"]) for r in runs}
        if len(runs) != count or len(observed) != count or observed != wanted:
            raise ValueError(f"{group}: incomplete or duplicated experiment design")
        for run in runs:
            key = (run["domain"], run["shots_per_class"], run["seed"])
            ids = tuple(sorted(run["target_label_ids"]))
            if paired.setdefault(key, ids) != ids:
                raise ValueError(f"{group}: label samples differ across protocols")
        reports[group] = verify(SimpleNamespace(
            data=root / "data", runs=folder, source=root / sources.get(group, group),
            model_path=None, config=None, device="cpu", out=None, predictions_only=True))
    for domain, seed in {(d, s) for d, _, s in paired}:
        budgets = sorted(k for d, k, s in paired if (d, s) == (domain, seed))
        for smaller, larger in zip(budgets, budgets[1:]):
            if not set(paired[(domain, smaller, seed)]) < set(paired[(domain, larger, seed)]):
                raise ValueError("Target-label budgets are not nested")
    report = {
        "status": "passed", "verification_mode": "saved_predictions",
        "verified_final_models": sum(r["verified_runs"] for r in reports.values()),
        "core_code_sha256": training_hash, "data_sha256": data_hash,
        "cross_protocol_paired_budget_groups": len(paired), "groups": reports,
        "scope": "Recomputed all six final-model groups from saved predictions; checked frozen code, complete designs, thresholds, labels, full and matched metrics, source baselines, and paired budgets.",
        "limitations": "This does not reconstruct model predictions or verify omitted weights. Historical full-checkpoint reports are retained separately. Sequential, mechanism, export, and second-test evidence is included but is outside this command's numerical recomputation scope."
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified {report['verified_final_models']} saved final-model records. Report: {args.out}")


if __name__ == "__main__":
    main()
