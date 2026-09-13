"""Matched two-fold model selection using only the same 2K target labels.

    This is a separate, fully logged protocol from fixed-recipe experiments.
    Each method gets the same three learning-rate multipliers and two folds.
    The source training checkpoint is reused; source tests never select a model.
"""
from __future__ import annotations
import argparse
import copy
import gc
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from .data import sha256
from .model import Classifier, tokenizer_from
from .protocol import hash_json, metrics, sample_budget
from .training import adapt, encode, predict, seed_everything
from .run import code_hash, evaluate, write_json


def run(args):
    cfg = json.loads(args.config.read_text())
    manifest = json.loads((args.data / "manifest.json").read_text())
    assert sha256(args.data / "records.jsonl") == manifest["records_sha256"]
    parent_plan = json.loads((args.single / "frozen_plan.json").read_text())
    assert parent_plan["configuration_sha256"] == hash_json(cfg)
    rows = [json.loads(s) for s in (args.data / "records.jsonl").read_text().splitlines()]
    device = args.device
    args.out.mkdir(parents=True, exist_ok=True)
    plan = {"base_configuration": cfg, "shots": [10, 80], "multipliers": [.25, 1., 4.],
            "folds": 2, "seed_list": cfg["seeds"], "methods": cfg["methods"],
            "selection": "mean held-out macro-F1 at .5 within 2K budget; ties prefer multiplier 1 then smaller multiplier",
            "final_training": "all 2K examples; fixed 80 steps; chosen learning-rate multiplier",
            "data_hash": manifest["records_sha256"], "core_code_hash": code_hash(),
            "cv_code_hash": sha256(__file__), "protocol": "strict_budget_cross_validation"}
    if (args.out / "frozen_plan.json").exists():
        assert json.loads((args.out / "frozen_plan.json").read_text()) == plan
    else:
        write_json(args.out / "frozen_plan.json", plan)
    tokenizer = tokenizer_from(args.model_path)
    test_rows = {d: [r for r in rows if r["domain"] == d and r["split"] == "test"] for d in ["source"] + cfg["domains"]}
    test = {d: encode(tokenizer, rs, cfg["max_length"]) for d, rs in test_rows.items()}
    for seed in cfg["seeds"]:
        info = json.loads((args.single / f"source_seed{seed}" / "source.json").read_text())
        source = Classifier(args.model_path, pretrained=True)
        source.load_state_dict(torch.load(args.single / f"source_seed{seed}" / "model.pt", map_location="cpu", weights_only=True))
        fisher = torch.load(args.single / f"source_seed{seed}" / "fisher.pt", map_location="cpu", weights_only=False)
        source.to(device)
        before, _ = evaluate(source, test["source"], test_rows["source"], "source", info["threshold"], manifest["matched_test_ids"]["source"], device)
        source.cpu()
        for domain in cfg["domains"]:
            pool = [r for r in rows if r["domain"] == domain and r["split"] == "train"]
            for shots in plan["shots"]:
                budget = sample_budget(pool, shots, seed, domain)
                folds = list(StratifiedKFold(n_splits=2, shuffle=True, random_state=seed).split(budget, [r["label"] for r in budget]))
                for method in cfg["methods"]:
                    identifier = f"{domain}_k{shots}_s{seed}_{method}"
                    result_path = args.out / "runs" / f"{identifier}.json"
                    if result_path.exists(): continue
                    candidates = []
                    for multiplier in plan["multipliers"]:
                        candidate_cfg = copy.deepcopy(cfg)
                        for key in ["module_lr", "calibration_lr", "full_lr"]:
                            candidate_cfg[key] *= multiplier
                        fold_logs, scores = [], []
                        for fold, (training_indices, validation_indices) in enumerate(folds):
                            seed_everything(seed + fold)
                            model = copy.deepcopy(source)
                            model.add_domain(domain, method, candidate_cfg)
                            train_rows = [budget[i] for i in training_indices]
                            valid_rows = [budget[i] for i in validation_indices]
                            train = encode(tokenizer, train_rows, cfg["max_length"])
                            valid = encode(tokenizer, valid_rows, cfg["max_length"])
                            training_info = adapt(model, train, domain, method, candidate_cfg, fisher, device)
                            probabilities = predict(model, valid, domain, device)
                            score = metrics(valid["labels"].numpy(), probabilities, .5)["macro_f1"]
                            scores.append(score)
                            fold_logs.append({"fold": fold, "train_ids": [r["id"] for r in train_rows],
                                              "validation_ids": [r["id"] for r in valid_rows], "macro_f1": score,
                                              "seconds": training_info["seconds"]})
                            del model
                        candidates.append({"multiplier": multiplier, "mean_cv_f1": float(np.mean(scores)), "folds": fold_logs})
                    selected = max(candidates, key=lambda c: (c["mean_cv_f1"], -abs(np.log(c["multiplier"])), -c["multiplier"]))
                    selected_cfg = copy.deepcopy(cfg)
                    for key in ["module_lr", "calibration_lr", "full_lr"]:
                        selected_cfg[key] *= selected["multiplier"]
                    seed_everything(seed)
                    model = copy.deepcopy(source)
                    model.add_domain(domain, method, selected_cfg)
                    training_info = adapt(model, encode(tokenizer, budget, cfg["max_length"]), domain, method, selected_cfg, fisher, device)
                    target, pred = evaluate(model, test[domain], test_rows[domain], domain, .5, manifest["matched_test_ids"][domain], device)
                    after, source_pred = evaluate(model, test["source"], test_rows["source"], "source", info["threshold"], manifest["matched_test_ids"]["source"], device)
                    checkpoint = args.out / "checkpoints" / f"{identifier}.pt"
                    checkpoint.parent.mkdir(exist_ok=True)
                    state = model.state_dict()
                    torch.save({"parameters": {n: state[n].detach().cpu() for n in training_info["trainable_names"]},
                                "domain": domain, "method": method, "source_seed": seed}, checkpoint)
                    write_json(args.out / "predictions" / f"{identifier}_target.json", pred)
                    write_json(args.out / "predictions" / f"{identifier}_source.json", source_pred)
                    training_info.update(candidate_count=3, fold_training_count=6,
                                         target_selection_labels_within_budget=2*shots,
                                         selected_multiplier=selected["multiplier"],
                                         selection_seconds=sum(f["seconds"] for c in candidates for f in c["folds"]))
                    result = {"run_id": identifier, "protocol": "strict_budget_cross_validation", "domain": domain,
                              "shots_per_class": shots, "seed": seed, "method": method,
                              "target_label_total": 2*shots, "target_label_ids": [r["id"] for r in budget],
                              "selected_configuration": selected_cfg, "selection_candidates": candidates,
                              "target": target, "source_before": before, "source_after": after,
                              "delta_source_pp": after["all"]["macro_f1"] - before["all"]["macro_f1"],
                              "threshold_target": .5, "threshold_source": info["threshold"], "training": training_info,
                              "checkpoint": str(checkpoint.relative_to(args.out)), "checkpoint_sha256": sha256(checkpoint)}
                    write_json(result_path, result)
                    print(f"CV {identifier}: selected {selected['multiplier']}, target={target['all']['macro_f1']:.2f}", flush=True)
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
