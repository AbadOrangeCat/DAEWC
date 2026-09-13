"""Post-fit evaluation on target partitions withheld from all earlier modelling.

These are the original target development partitions, now used only as a second
test. Every prespecified compact-model baseline is reported. No scores from this
evaluation enter selection, thresholds, or further training.
"""
import argparse
import copy
import json
from pathlib import Path
import torch
from .data import sha256
from .model import Classifier, tokenizer_from
from .protocol import metrics
from .training import encode, predict
from .run import write_json


def run(args):
    torch.set_num_threads(2)
    rows = [json.loads(s) for s in (args.data / "records.jsonl").read_text().splitlines()]
    groups = [("fixed_reference", "local", "local", "local.json"),
              ("fixed_low_footprint", "low_footprint", "low_footprint", "low_footprint.json"),
              ("cv_reference", "budget_cv", "local", "local.json"),
              ("cv_low_footprint", "low_footprint_cv", "low_footprint", "low_footprint.json")]
    plan = {"evaluation_partition": "original target dev; post-fit evaluation only",
            "shots": 80, "groups": groups, "threshold": .5, "seeds": [42,43,44],
            "domains": ["health", "public_affairs", "entertainment"],
            "selection": "none; all prespecified groups reported", "code_sha256": sha256(__file__),
            "data_sha256": sha256(args.data / "records.jsonl")}
    args.out.mkdir(parents=True, exist_ok=True)
    # JSON converts tuples to lists, so normalize before comparison.
    plan = json.loads(json.dumps(plan))
    if (args.out / "frozen_plan.json").exists():
        assert json.loads((args.out / "frozen_plan.json").read_text()) == plan
    else:
        write_json(args.out / "frozen_plan.json", plan)
    if args.freeze_only:
        print("Secondary-test plan frozen; no evaluation performed.")
        return
    tokenizer = tokenizer_from(args.model_path)
    test_rows = {d: [r for r in rows if r["domain"] == d and r["split"] == "dev"] for d in plan["domains"]}
    test = {d: encode(tokenizer, rs, 128) for d, rs in test_rows.items()}
    for group, run_dir, source_dir, config_name in groups:
        cfg = json.loads((args.configs / config_name).read_text())
        expected = len(cfg["methods"]) * 3 * 3
        run_files = [p for p in (args.artifacts / run_dir / "runs").glob("*_k80_*.json")
                     if json.loads(p.read_text())["method"] in cfg["methods"]]
        if len(run_files) != expected:
            raise ValueError(f"{group}: expected {expected} completed runs, found {len(run_files)}")
        for seed in plan["seeds"]:
            source = Classifier(args.model_path, pretrained=False)
            source.load_state_dict(torch.load(args.artifacts / source_dir / f"source_seed{seed}" / "model.pt", map_location="cpu", weights_only=True))
            for path in sorted(run_files):
                r = json.loads(path.read_text())
                if r["seed"] != seed: continue
                identifier = group + "_" + r["run_id"]
                result_path = args.out / "runs" / f"{identifier}.json"
                if result_path.exists(): continue
                checkpoint = args.artifacts / run_dir / r["checkpoint"]
                assert sha256(checkpoint) == r["checkpoint_sha256"]
                model = copy.deepcopy(source)
                model.add_domain(r["domain"], r["method"], cfg)
                model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["parameters"], strict=False)
                probabilities = predict(model, test[r["domain"]], r["domain"], "cpu", 64)
                scores = metrics(test[r["domain"]]["labels"].numpy(), probabilities, .5)
                predictions = [{"id": row["id"], "label": row["label"], "probability": p}
                               for row, p in zip(test_rows[r["domain"]], probabilities)]
                write_json(args.out / "predictions" / f"{identifier}.json", predictions)
                write_json(result_path, {"id": identifier, "group": group, "method": r["method"],
                                         "seed": seed, "domain": r["domain"], "shots_per_class": 80,
                                         "threshold": .5, "scores": scores, "test_n": len(predictions),
                                         "checkpoint_sha256": r["checkpoint_sha256"],
                                         "trainable_parameters": r["training"]["trainable_parameters"],
                                         "selection_or_training_performed": False})
                print(identifier, scores["macro_f1"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for field in ["data", "artifacts", "configs", "out"]: p.add_argument("--"+field, type=Path, required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--freeze-only", action="store_true")
    run(p.parse_args())
