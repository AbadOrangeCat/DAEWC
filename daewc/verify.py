"""Verify result arithmetic, label budgets, split integrity, and saved checkpoints."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from .data import sha256
from .model import Classifier, tokenizer_from
from .protocol import metrics
from .training import encode, predict


def verified_scores(predictions, rows, domain, threshold, matched_ids):
    """Check each prediction against the frozen protocol before scoring it."""
    expected = {r["id"]: r for r in rows if r["domain"] == domain and r["split"] == "test"}
    identifiers = [p["id"] for p in predictions]
    if len(identifiers) != len(set(identifiers)) or set(identifiers) != set(expected):
        raise ValueError(f"{domain}: duplicate, missing, or unexpected test identifiers")
    for prediction in predictions:
        if prediction["threshold"] != threshold:
            raise ValueError(f"{domain}: prediction threshold differs from the frozen threshold")
        probability = prediction["probability"]
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(f"{domain}: invalid probability")
        if prediction["label"] != expected[prediction["id"]]["label"]:
            raise ValueError(f"{domain}: prediction label differs from the prepared data")
        if prediction["origin_length_matched"] != (prediction["id"] in matched_ids):
            raise ValueError(f"{domain}: incorrect matched-subset membership")
    matched = [p for p in predictions if p["id"] in matched_ids]
    def score(records):
        return metrics([p["label"] for p in records], [p["probability"] for p in records], threshold)
    return {"all": score(predictions), "matched": score(matched) if matched else None,
            "matched_n": len(matched)}


def verify(args):
    rows = [json.loads(s) for s in (args.data / "records.jsonl").read_text().splitlines()]
    manifest = json.loads((args.data / "manifest.json").read_text())
    assert sha256(args.data / "records.jsonl") == manifest["records_sha256"]
    lookup = {r["id"]: r for r in rows}
    assert len(lookup) == len(rows)
    assert len({r["cluster"] for r in rows}) == len(rows)
    runs = [json.loads(p.read_text()) for p in (args.runs / "runs").glob("*.json")]
    source_root = args.source or args.runs
    sources = {}
    for seed in {r["seed"] for r in runs}:
        folder = source_root / f"source_seed{seed}"
        info = json.loads((folder / "source.json").read_text())
        predictions = json.loads((folder / "test_predictions.json").read_text())
        sources[seed] = (info["threshold"], verified_scores(
            predictions, rows, "source", info["threshold"], set(manifest["matched_test_ids"]["source"])))
    budgets = defaultdict(set)
    for r in runs:
        ids = r["target_label_ids"]
        assert len(ids) == len(set(ids)) == r["target_label_total"] == 2 * r["shots_per_class"]
        assert all(lookup[i]["split"] == "train" and lookup[i]["domain"] == r["domain"] for i in ids)
        assert sum(lookup[i]["label"] for i in ids) == r["shots_per_class"]
        budgets[(r["domain"], r["shots_per_class"], r["seed"])].add(tuple(sorted(ids)))
        assert r["threshold_target"] == .5
        source_threshold, source_scores = sources[r["seed"]]
        assert r["threshold_source"] == source_threshold, r["run_id"]
        assert r["source_before"] == source_scores, r["run_id"]
        assert r["training"]["target_dev_labels_used"] == 0
        assert not r["training"]["source_replay"] and not r["training"]["unlabeled_target"]
        assert r["training"]["optimizer_steps"] == 80
        assert r["training"]["trainable_parameters"] == sum(r["training"]["trainable_names"].values())
        assert sha256(args.runs / r["checkpoint"]) == r["checkpoint_sha256"]
        for role, domain, resultkey in [("source", "source", "source_after"), ("target", r["domain"], "target")]:
            preds = json.loads((args.runs / "predictions" / f"{r['run_id']}_{role}.json").read_text())
            threshold = source_threshold if role == "source" else .5
            verified = verified_scores(preds, rows, domain, threshold, set(manifest["matched_test_ids"][domain]))
            score = verified["all"]
            assert all(abs(score[k] - r[resultkey]["all"][k]) < 1e-9 for k in score)
            matched_ids = set(manifest['matched_test_ids'][domain])
            assert all(p['origin_length_matched'] == (p['id'] in matched_ids) for p in preds)
            matched = [p for p in preds if p['id'] in matched_ids]
            assert len(matched) == r[resultkey]['matched_n'] == len(matched_ids)
            matched_score = verified["matched"]
            assert all(abs(matched_score[k] - r[resultkey]['matched'][k]) < 1e-9 for k in matched_score)
        assert abs(r["delta_source_pp"] - (r["source_after"]["all"]["macro_f1"] - r["source_before"]["all"]["macro_f1"])) < 1e-9
        for candidate in r.get("selection_candidates", []):
            assert len(candidate["folds"]) == 2
            assert abs(candidate["mean_cv_f1"] - np.mean([f["macro_f1"] for f in candidate["folds"]])) < 1e-12
            validation_ids = []
            for fold in candidate["folds"]:
                train_ids, val_ids = set(fold["train_ids"]), set(fold["validation_ids"])
                assert not train_ids & val_ids and train_ids | val_ids == set(ids)
                validation_ids.extend(val_ids)
            assert len(validation_ids) == len(set(validation_ids)) == len(ids)
        if r.get("selection_candidates"):
            candidates = r["selection_candidates"]
            assert len(candidates) == 3 and {c["multiplier"] for c in candidates} == {.25, 1., 4.}
            selected = max(candidates, key=lambda c: (c["mean_cv_f1"], -abs(np.log(c["multiplier"])), -c["multiplier"]))
            assert r["training"]["selected_multiplier"] == selected["multiplier"]
            base_cfg = json.loads((args.runs / "frozen_plan.json").read_text())["base_configuration"]
            expected_cfg = dict(base_cfg)
            for key in ["module_lr", "calibration_lr", "full_lr"]:
                expected_cfg[key] *= selected["multiplier"]
            assert r["selected_configuration"] == expected_cfg
    assert all(len(v) == 1 for v in budgets.values())
    reproduced = []
    if args.model_path:
        tokenizer = tokenizer_from(args.model_path)
        cfg = json.loads(args.config.read_text())
        # Reconstruct a complete test partition per method. Prefer the smallest
        # partition so this parameter-scope check remains practical for BERT-base.
        selected = {}
        test_sizes = {domain: sum(x['domain'] == domain and x['split'] == 'test' for x in rows)
                      for domain in {r['domain'] for r in runs}}
        for r in sorted(runs, key=lambda r: (test_sizes[r['domain']], r['run_id'])):
            selected.setdefault(r["method"], r)
        for method, r in selected.items():
            model = Classifier(args.model_path, pretrained=False)
            state = torch.load(args.source / f"source_seed{r['seed']}" / "model.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.add_domain(r["domain"], method, cfg)
            checkpoint = torch.load(args.runs / r["checkpoint"], map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["parameters"], strict=False)
            subset = [x for x in rows if x["domain"] == r["domain"] and x["split"] == "test"]
            encoded = encode(tokenizer, subset, cfg["max_length"])
            prob = predict(model.to(args.device), encoded, r["domain"], args.device)
            expected = json.loads((args.runs / "predictions" / f"{r['run_id']}_target.json").read_text())
            assert [p["id"] for p in expected] == encoded["ids"]
            error = float(np.max(np.abs(np.asarray(prob) - np.asarray([p["probability"] for p in expected]))))
            assert error < 1e-5, (method, error)
            reproduced.append({"run_id": r["run_id"], "max_probability_error": error})
    result = {"verified_runs": len(runs), "paired_budget_groups": len(budgets), "checkpoint_reconstructions": reproduced,
              "checkpoint_reconstruction_device": args.device if args.model_path else None,
              "checks": "data and checkpoint SHA-256; unique prediction IDs and valid probabilities; frozen per-example thresholds; independently recomputed source baseline; split membership; strict label counts; paired samples; metric recomputation; CV fold isolation, candidate scores and selected configuration"}
    print(json.dumps(result, indent=2))
    if args.out: args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for field in ["data", "runs"]: p.add_argument("--"+field, type=Path, required=True)
    for field in ["source", "config", "out"]: p.add_argument("--"+field, type=Path)
    p.add_argument("--model-path")
    p.add_argument("--device", default="mps")
    verify(p.parse_args())
