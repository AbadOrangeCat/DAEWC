"""A complete predeclared calibration-rate and Fisher-weight grid, not selection."""
import argparse
import copy
import gc
import itertools
import json
from pathlib import Path
import torch
from .data import sha256
from .model import Classifier, tokenizer_from
from .protocol import sample_budget
from .training import adapt, encode, seed_everything
from .run import evaluate, write_json


def run(args):
    cfg = json.loads(args.config.read_text())
    manifest = json.loads((args.data / "manifest.json").read_text())
    rows = [json.loads(s) for s in (args.data / "records.jsonl").read_text().splitlines()]
    assert sha256(args.data / "records.jsonl") == manifest["records_sha256"]
    rates, weights = [1e-4, 1e-3, 1e-2], [0., 100., 10000.]
    plan = {"base_configuration": cfg, "calibration_rates": rates, "ewc_weights": weights,
            "shots_per_class": 80, "code_sha256": sha256(__file__),
            "purpose": "Report the entire grid; do not select or rename a winning DAEWC recipe"}
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "frozen_plan.json").exists():
        assert json.loads((args.out / "frozen_plan.json").read_text()) == plan
    else: write_json(args.out / "frozen_plan.json", plan)
    tokenizer = tokenizer_from(args.model_path)
    test_rows = {d: [r for r in rows if r["domain"] == d and r["split"] == "test"] for d in ["source"] + cfg["domains"]}
    test = {d: encode(tokenizer, rs, cfg["max_length"]) for d, rs in test_rows.items()}
    for seed in cfg["seeds"]:
        info = json.loads((args.single / f"source_seed{seed}" / "source.json").read_text())
        source = Classifier(args.model_path)
        source.load_state_dict(torch.load(args.single / f"source_seed{seed}" / "model.pt", map_location="cpu", weights_only=True))
        fisher = torch.load(args.single / f"source_seed{seed}" / "fisher.pt", map_location="cpu", weights_only=False)
        source.to(args.device)
        before, _ = evaluate(source, test["source"], test_rows["source"], "source", info["threshold"], [], args.device)
        source.cpu()
        for domain in cfg["domains"]:
            budget = sample_budget([r for r in rows if r["domain"] == domain and r["split"] == "train"], 80, seed, domain)
            train = encode(tokenizer, budget, cfg["max_length"])
            for rate, weight in itertools.product(rates, weights):
                identifier = f"s{seed}_{domain}_lr{rate:g}_lambda{weight:g}"
                path = args.out / "runs" / f"{identifier}.json"
                if path.exists(): continue
                current = copy.deepcopy(cfg)
                current.update(calibration_lr=rate, ewc_lambda=weight)
                seed_everything(seed)
                model = copy.deepcopy(source)
                model.add_domain(domain, "daewc", current)
                training = adapt(model, train, domain, "daewc", current, fisher, args.device)
                target, pred = evaluate(model, test[domain], test_rows[domain], domain, .5, manifest["matched_test_ids"][domain], args.device)
                after, src_pred = evaluate(model, test["source"], test_rows["source"], "source", info["threshold"], [], args.device)
                state = model.state_dict()
                checkpoint = args.out / "checkpoints" / f"{identifier}.pt"
                checkpoint.parent.mkdir(exist_ok=True)
                torch.save({"parameters": {n: state[n].detach().cpu() for n in training["trainable_names"]},
                            "domain": domain, "method": "daewc", "source_seed": seed}, checkpoint)
                write_json(args.out / "predictions" / f"{identifier}.json", {"target": pred, "source": src_pred})
                write_json(path, {"id": identifier, "seed": seed, "domain": domain, "calibration_lr": rate,
                                  "lambda": weight, "configuration": current, "label_ids": [r["id"] for r in budget],
                                  "target": target, "source_before": before, "source_after": after,
                                  "delta_source_pp": after["all"]["macro_f1"]-before["all"]["macro_f1"],
                                  "checkpoint": str(checkpoint.relative_to(args.out)), "checkpoint_sha256": sha256(checkpoint),
                                  "training": training})
                print(identifier, target["all"]["macro_f1"], after["all"]["macro_f1"]-before["all"]["macro_f1"], flush=True)
                del model
                gc.collect()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for field in ["config", "data", "single", "out"]: p.add_argument("--"+field, type=Path, required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--device", default="mps")
    run(p.parse_args())
