"""Run a target-only configuration change from exact existing source checkpoints.

The original source fit is reused by checksum, not repeated. This prevents tiny
device-dependent differences in source retraining from entering paired variants.
"""
import argparse
import json
import shutil
from pathlib import Path
from .data import sha256
from .protocol import hash_json
from .run import run, write_json


def prepare(args):
    cfg = json.loads(args.config.read_text())
    source_plan = json.loads((args.source / "frozen_plan.json").read_text())
    source_cfg = source_plan["configuration"]
    fields = ["model_id", "model_revision", "initialization", "max_length", "batch_size",
              "eval_batch_size", "source_epochs", "source_patience", "source_lr",
              "weight_decay", "gradient_clip", "fisher_samples"]
    if any(cfg[k] != source_cfg[k] for k in fields):
        raise ValueError("Source training, initialization, tokenization or Fisher settings differ")
    data_hash = sha256(args.data / "records.jsonl")
    if data_hash != source_plan["data_sha256"]:
        raise ValueError("Source data differ")
    provenance = {"source_plan_sha256": sha256(args.source / "frozen_plan.json"),
                  "original_source_configuration_sha256": hash_json(source_cfg),
                  "target_configuration_sha256": hash_json(cfg), "data_sha256": data_hash,
                  "source_fields_verified": fields, "importer_sha256": sha256(__file__),
                  "checkpoints": {}}
    for seed in cfg["seeds"]:
        origin = args.source / f"source_seed{seed}"
        info = json.loads((origin / "source.json").read_text())
        if sha256(origin / "model.pt") != info["checkpoint_sha256"]:
            raise ValueError("Original source checkpoint checksum differs")
        dest = args.out / f"source_seed{seed}"
        dest.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for name in ["model.pt", "fisher.pt"]:
            digest = sha256(origin / name)
            if (dest / name).exists():
                if sha256(dest / name) != digest:
                    raise ValueError("Destination contains a different source; use an empty output directory")
            else:
                shutil.copy2(origin / name, dest / name)
            hashes[name] = digest
        info["reused_source_configuration_sha256"] = info["config_hash"]
        info["config_hash"] = hash_json(cfg)
        info["source_retrained_for_this_configuration"] = False
        write_json(dest / "source.json", info)
        provenance["checkpoints"][str(seed)] = hashes
    write_json(args.out / "source_import.json", provenance)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for field in ["config", "source", "data", "out"]:
        p.add_argument("--" + field, type=Path, required=True)
    p.add_argument("--model-path")
    p.add_argument("--device")
    args = p.parse_args()
    prepare(args)
    run(args)
