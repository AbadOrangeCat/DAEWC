"""Check the objective condition of the conditional stability proposition.

This measures the actual penalty and target loss. It does not estimate or certify
the integrated source-gradient bound, and therefore makes no certified F1 claim.
"""
import argparse
import copy
import csv
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from .model import Classifier, tokenizer_from
from .training import batch, encode
from .run import write_json


@torch.no_grad()
def cross_entropy(model, data, domain):
    model.eval()
    total = 0.
    for start in range(0, len(data["labels"]), 32):
        b = batch(data, slice(start, start + 32), "cpu")
        total += float(F.cross_entropy(model(b["input_ids"], b["attention_mask"], domain), b["labels"], reduction="sum"))
    return total / len(data["labels"])


def run(args):
    torch.set_num_threads(2)
    cfg = json.loads(args.config.read_text())
    rows = {r["id"]: r for r in (json.loads(s) for s in (args.data / "records.jsonl").read_text().splitlines())}
    runs = [json.loads(p.read_text()) for p in (args.runs / "runs").glob("*.json")]
    runs = [r for r in runs if r["method"] == "daewc"]
    tokenizer = tokenizer_from(args.model_path)
    result = []
    for seed in cfg["seeds"]:
        source = Classifier(args.model_path, pretrained=False)
        source.load_state_dict(torch.load(args.runs / f"source_seed{seed}" / "model.pt", map_location="cpu", weights_only=True))
        fisher = torch.load(args.runs / f"source_seed{seed}" / "fisher.pt", map_location="cpu", weights_only=False)
        for r in [r for r in runs if r["seed"] == seed]:
            data = encode(tokenizer, [rows[i] for i in r["target_label_ids"]], cfg["max_length"])
            j0 = cross_entropy(source, data, "source")
            model = copy.deepcopy(source)
            model.add_domain(r["domain"], "daewc", cfg)
            state = torch.load(args.runs / r["checkpoint"], map_location="cpu", weights_only=True)
            model.load_state_dict(state["parameters"], strict=False)
            ce = cross_entropy(model, data, r["domain"])
            quadratic = 0.
            for n, p in model.named_parameters():
                if n in model.calibration_names():
                    diagonal = cfg["ewc_lambda"] * (fisher["fisher"][n] + cfg["fisher_damping"]) + cfg["proximity_alpha"]
                    quadratic += float((diagonal * (p.detach() - fisher["star"][n]).square()).sum())
            result.append({"run_id": r["run_id"], "initial_target_cross_entropy": j0,
                           "final_target_cross_entropy": ce, "weighted_drift_squared": quadratic,
                           "final_objective": ce + .5*quadratic, "objective_condition_holds": ce + .5*quadratic <= j0,
                           "drift_bound_holds": quadratic <= 2*j0,
                           "source_gradient_bound_certified": False})
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "stability_diagnostics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    summary = {"runs": len(result), "objective_condition_satisfied": sum(r["objective_condition_holds"] for r in result),
               "drift_bound_satisfied": sum(r["drift_bound_holds"] for r in result),
               "source_gradient_bound_certified": False, "evaluation": "deterministic CPU forward pass, dropout disabled"}
    write_json(args.out / "stability_diagnostics.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for field in ["config", "data", "runs", "out"]: p.add_argument("--"+field, type=Path, required=True)
    p.add_argument("--model-path", required=True)
    run(p.parse_args())
