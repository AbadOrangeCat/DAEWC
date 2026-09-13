"""Run strict-budget experiments and record complete auditable artifacts."""
from __future__ import annotations
import argparse
import copy
import gc
import importlib.metadata
import itertools
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from .data import sha256
from .model import Classifier, tokenizer_from
from .protocol import hash_json, metrics, retention, sample_budget
from .training import adapt, empirical_fisher, encode, predict, seed_everything, source_train


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    temp.replace(path)


def code_hash():
    root = Path(__file__).parent
    return hash_json({name: sha256(root / name) for name in ["data.py", "model.py", "protocol.py", "training.py", "run.py"]})


def load_environment():
    names = ["torch", "transformers", "numpy", "scikit-learn", "datasketch"]
    return {"platform": platform.platform(), "python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name) for name in names}}


def get_source(seed, cfg, model_path, data, out, device):
    folder = out / f"source_seed{seed}"
    info_path = folder / "source.json"
    model = Classifier(model_path=model_path, pretrained=cfg["initialization"] == "pretrained")
    if info_path.exists():
        info = json.loads(info_path.read_text())
        if info["config_hash"] != hash_json(cfg):
            raise ValueError("Source cache configuration differs; use a new output directory")
        model.load_state_dict(torch.load(folder / "model.pt", map_location="cpu", weights_only=True))
        fisher = torch.load(folder / "fisher.pt", map_location="cpu", weights_only=False)
        return model, fisher, info
    threshold, source_info = source_train(model, data["source"]["train"], data["source"]["dev"], cfg, device)
    fisher = empirical_fisher(model, data["source"]["train"], model.shared_names(),
                              cfg["fisher_samples"], seed + 1000, device)
    folder.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, folder / "model.pt")
    torch.save(fisher, folder / "fisher.pt")
    info = {"seed": seed, "config_hash": hash_json(cfg), "threshold": threshold, "source_training": source_info,
            "fisher_seconds": fisher["seconds"], "fisher_ids": fisher["sample_ids"],
            "checkpoint_sha256": sha256(folder / "model.pt"),
            "calibration_names": sorted(model.calibration_names()),
            "calibration_parameters": sum(p.numel() for n, p in model.named_parameters() if n in model.calibration_names()),
            "total_parameters": sum(p.numel() for p in model.parameters())}
    write_json(info_path, info)
    return model.cpu(), fisher, info


def evaluate(model, encoded, rows, domain, threshold, matched, device):
    probabilities = predict(model, encoded, domain, device)
    all_scores = metrics(encoded["labels"].numpy(), probabilities, threshold)
    wanted = set(matched)
    subset = [i for i, r in enumerate(rows) if r["id"] in wanted]
    matched_scores = (metrics([rows[i]["label"] for i in subset], [probabilities[i] for i in subset], threshold)
                      if subset else None)
    predictions = [{"id": row["id"], "label": row["label"], "probability": p, "threshold": threshold,
                    "origin_length_matched": row["id"] in wanted} for row, p in zip(rows, probabilities)]
    return {"all": all_scores, "matched": matched_scores, "matched_n": len(subset)}, predictions


def run(args):
    cfg = json.loads(args.config.read_text())
    out, data_root = args.out.resolve(), args.data.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((data_root / "manifest.json").read_text())
    if sha256(data_root / "records.jsonl") != manifest["records_sha256"]:
        raise ValueError("Prepared data do not match the saved manifest")
    rows = [json.loads(line) for line in (data_root / "records.jsonl").read_text().splitlines()]
    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    freeze = {"configuration": cfg, "configuration_sha256": hash_json(cfg), "data_sha256": manifest["records_sha256"],
              "code_sha256": code_hash(), "device": device, "environment": load_environment(),
              "selection_rule": "all specified runs reported; no choice using any test result",
              "timestamp_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()}
    if (out / "frozen_plan.json").exists():
        previous = json.loads((out / "frozen_plan.json").read_text())
        for key in ["configuration_sha256", "data_sha256", "code_sha256"]:
            if previous[key] != freeze[key]:
                raise ValueError(f"Resume refused: {key} changed; create a new output directory")
    else:
        write_json(out / "frozen_plan.json", freeze)
    model_path = args.model_path or snapshot_download(cfg["model_id"], revision=cfg["model_revision"],
                                                      allow_patterns=["config.json", "vocab.txt", "model.safetensors"])
    tokenizer = tokenizer_from(model_path)
    grouped, data = {}, {}
    token_audit = {}
    for domain in ["source"] + cfg["domains"]:
        grouped[domain], data[domain] = {}, {}
        for split in ["train", "dev", "test"]:
            # Target dev files are deliberately not tokenized or sent to training.
            if split == "dev" and domain != "source":
                continue
            subset = [r for r in rows if r["domain"] == domain and r["split"] == split]
            grouped[domain][split] = subset
            data[domain][split] = encode(tokenizer, subset, cfg["max_length"])
        tokens = data[domain]["train"]["input_ids"]
        mask = data[domain]["train"]["attention_mask"].bool()
        special = (tokens == tokenizer.cls_token_id) | (tokens == tokenizer.sep_token_id)
        denominator = int((mask & ~special).sum())
        token_audit[domain] = {"unk_tokens": int((tokens == tokenizer.unk_token_id).sum()),
                               "content_tokens": denominator, "unknown_token_rate": float((tokens == tokenizer.unk_token_id).sum()) / max(1, denominator)}
    write_json(out / "token_audit.json", token_audit)
    runs = []
    for seed in cfg["seeds"]:
        seed_everything(seed)
        source, fisher, source_info = get_source(seed, cfg, model_path, data, out, device)
        source.to(device)
        source_scores, source_predictions = evaluate(source, data["source"]["test"], grouped["source"]["test"],
                                                     "source", source_info["threshold"], manifest["matched_test_ids"]["source"], device)
        write_json(out / f"source_seed{seed}" / "test_predictions.json", source_predictions)
        zero = {"source": source_scores}
        for domain in cfg["domains"]:
            probabilities = predict(source, data[domain]["test"], "source", device)
            zero[domain] = metrics(data[domain]["test"]["labels"].numpy(), probabilities, .5)
        write_json(out / f"source_seed{seed}" / "zero_shot.json", zero)
        source.cpu()
        for domain, shots in itertools.product(cfg["domains"], cfg["shots"]):
            try:
                budget = sample_budget(grouped[domain]["train"], shots, seed, domain)
            except ValueError as error:
                write_json(out / f"unavailable_{domain}_k{shots}_s{seed}.json", {"reason": str(error)})
                continue
            train = encode(tokenizer, budget, cfg["max_length"])
            methods = list(cfg["methods"])
            if shots in cfg["ablation_shots"]:
                methods += cfg["ablation_methods"]
            for method in methods:
                run_id = f"{domain}_k{shots}_s{seed}_{method}"
                result_path = out / "runs" / f"{run_id}.json"
                if result_path.exists():
                    runs.append(json.loads(result_path.read_text()))
                    continue
                seed_everything(seed)
                model = copy.deepcopy(source)
                model.add_domain(domain, method, cfg)
                info = adapt(model, train, domain, method, cfg, fisher, device)
                target_scores, target_predictions = evaluate(model, data[domain]["test"], grouped[domain]["test"], domain, .5,
                                                              manifest["matched_test_ids"][domain], device)
                after, after_predictions = evaluate(model, data["source"]["test"], grouped["source"]["test"], "source", source_info["threshold"],
                                                      manifest["matched_test_ids"]["source"], device)
                # Freeze inference state before recording metrics. Save a source-relative
                # checkpoint containing every trainable parameter of this procedure.
                checkpoint = out / "checkpoints" / f"{run_id}.pt"
                checkpoint.parent.mkdir(exist_ok=True)
                state = model.state_dict()
                torch.save({"parameters": {name: state[name].detach().cpu() for name in info["trainable_names"]},
                            "domain": domain, "method": method, "source_seed": seed}, checkpoint)
                pred_dir = out / "predictions"
                write_json(pred_dir / f"{run_id}_target.json", target_predictions)
                write_json(pred_dir / f"{run_id}_source.json", after_predictions)
                result = {"run_id": run_id, "domain": domain, "shots_per_class": shots, "seed": seed, "method": method,
                          "config_hash": hash_json(cfg), "data_hash": manifest["records_sha256"],
                          "target_label_ids": [r["id"] for r in budget], "target_label_total": len(budget),
                          "target": target_scores, "source_before": source_scores, "source_after": after,
                          "delta_source_pp": retention(source_scores["all"]["macro_f1"], after["all"]["macro_f1"]),
                          "threshold_target": .5, "threshold_source": source_info["threshold"],
                          "checkpoint": str(checkpoint.relative_to(out)), "checkpoint_sha256": sha256(checkpoint),
                          "training": info}
                write_json(result_path, result)
                runs.append(result)
                print(f"{run_id}: target={target_scores['all']['macro_f1']:.2f}, source change={result['delta_source_pp']:.3f} pp, {info['seconds']:.1f}s", flush=True)
                del model
                gc.collect()
                if device == "mps":
                    torch.mps.empty_cache()
        del source, fisher
    write_json(out / "run_index.json", [{k: r[k] for k in ["run_id", "checkpoint", "checkpoint_sha256"]} for r in runs])
    print(f"Completed {len(runs)} runs", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "mps", "cuda"])
    p.add_argument("--model-path", type=str)
    run(p.parse_args())


if __name__ == "__main__":
    main()
