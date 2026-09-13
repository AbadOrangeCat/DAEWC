"""All six orders of three domains, with an independently recorded score matrix."""
from __future__ import annotations
import argparse
import copy
import gc
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from .data import sha256
from .model import Classifier, tokenizer_from
from .protocol import hash_json, sample_budget
from .training import adapt, encode, predict, seed_everything
from .run import code_hash, evaluate, load_environment, write_json


def run(args):
    cfg = json.loads(args.config.read_text())
    data_manifest = json.loads((args.data / "manifest.json").read_text())
    assert sha256(args.data / "records.jsonl") == data_manifest["records_sha256"]
    primary_plan = json.loads((args.single / "frozen_plan.json").read_text())
    assert primary_plan["configuration_sha256"] == hash_json(cfg)
    assert primary_plan["code_sha256"] == code_hash()
    rows = [json.loads(s) for s in (args.data / "records.jsonl").read_text().splitlines()]
    device = args.device
    args.out.mkdir(parents=True, exist_ok=True)
    plan = {"configuration": cfg, "data_hash": data_manifest["records_sha256"], "core_code_hash": code_hash(),
            "sequential_code_hash": sha256(__file__), "orders": list(itertools.permutations(cfg["domains"])),
            "fisher_policy": "source Fisher and source anchor remain fixed across all stages",
            "old_domain_modules": "frozen; shared calibration can still change predictions",
            "device": device, "environment": load_environment()}
    if (args.out / "frozen_plan.json").exists():
        assert json.loads((args.out / "frozen_plan.json").read_text()) == plan
    else:
        write_json(args.out / "frozen_plan.json", plan)
    tokenizer = tokenizer_from(args.model_path)
    test_rows = {d: [r for r in rows if r["domain"] == d and r["split"] == "test"] for d in ["source"] + cfg["domains"]}
    test = {d: encode(tokenizer, rs, cfg["max_length"]) for d, rs in test_rows.items()}
    for seed in cfg["seeds"]:
        info = json.loads((args.single / f"source_seed{seed}" / "source.json").read_text())
        source = Classifier(args.model_path, pretrained=cfg["initialization"] == "pretrained")
        source.load_state_dict(torch.load(args.single / f"source_seed{seed}" / "model.pt", map_location="cpu", weights_only=True))
        fisher = torch.load(args.single / f"source_seed{seed}" / "fisher.pt", map_location="cpu", weights_only=False)
        for order in itertools.permutations(cfg["domains"]):
            for method in cfg["sequential_methods"]:
                identifier = f"s{seed}_{method}_{'-'.join(order)}"
                path = args.out / "runs" / f"{identifier}.json"
                if path.exists():
                    continue
                seed_everything(seed)
                model = copy.deepcopy(source).to(device)
                matrix, stages = [], []
                initial = {}
                from .protocol import metrics
                for domain in ["source"] + list(order):
                    p = predict(model, test[domain], "source", device)
                    initial[domain] = metrics(test[domain]["labels"].numpy(), p,
                                               info["threshold"] if domain == "source" else .5)["macro_f1"]
                matrix.append(initial)
                for step, domain in enumerate(order, start=1):
                    # The same labels are used at this K and seed, regardless of order.
                    budget = sample_budget([r for r in rows if r["domain"] == domain and r["split"] == "train"],
                                           cfg["sequential_shots"], seed, domain)
                    train = encode(tokenizer, budget, cfg["max_length"])
                    model.add_domain(domain, method, cfg)
                    training = adapt(model, train, domain, method, cfg, fisher, device)
                    row, predictions = {}, {}
                    for seen in ["source"] + list(order[:step]):
                        scores, pred = evaluate(model, test[seen], test_rows[seen], seen,
                                                 info["threshold"] if seen == "source" else .5,
                                                 data_manifest["matched_test_ids"][seen], device)
                        row[seen] = scores["all"]["macro_f1"]
                        predictions[seen] = pred
                    matrix.append(row)
                    drift = sum(float((p.detach().cpu() - fisher["star"][n]).square().sum())
                                for n, p in model.named_parameters() if n in model.calibration_names())**.5
                    checkpoint = args.out / "checkpoints" / f"{identifier}_stage{step}.pt"
                    checkpoint.parent.mkdir(exist_ok=True)
                    # Incremental checkpoint; stage t depends on source and stages <t.
                    state = model.state_dict()
                    torch.save({"parameters": {n: state[n].detach().cpu() for n in training["trainable_names"]},
                                "stage": step, "order": order, "method": method}, checkpoint)
                    write_json(args.out / "predictions" / f"{identifier}_stage{step}.json", predictions)
                    stages.append({"stage": step, "domain": domain, "label_ids": [r["id"] for r in budget],
                                   "training": training, "shared_calibration_l2_from_source": drift,
                                   "checkpoint": str(checkpoint.relative_to(args.out)), "checkpoint_sha256": sha256(checkpoint)})
                # Older targets only, and an explicitly separate source-inclusive BWT.
                differences = [matrix[-1][d] - matrix[j+1][d] for j, d in enumerate(order[:-1])]
                source_change = matrix[-1]["source"] - matrix[0]["source"]
                result = {"id": identifier, "seed": seed, "method": method, "order": order,
                          "shots_per_class": cfg["sequential_shots"], "matrix": matrix, "stages": stages,
                          "bwt_older_targets_pp": float(np.mean(differences)),
                          "bwt_including_source_pp": float(np.mean([source_change] + differences)),
                          "delta_source_pp": source_change,
                          "final_target_macro_f1": float(np.mean([matrix[-1][d] for d in order]))}
                write_json(path, result)
                print(f"Sequence {identifier}: BWT={result['bwt_older_targets_pp']:.3f} pp", flush=True)
                del model
                gc.collect()
                if device == "mps": torch.mps.empty_cache()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--single", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    run(p.parse_args())
