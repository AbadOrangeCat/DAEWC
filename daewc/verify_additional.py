"""Independently check sequential, mechanism, and second-test evidence."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from .data import sha256
from .protocol import metrics, sample_budget
from .run import write_json


def run(args):
    root = args.artifacts
    rows = [json.loads(s) for s in (args.data / "records.jsonl").read_text().splitlines()]
    lookup = {r["id"]: r for r in rows}
    domains = ["health", "public_affairs", "entertainment"]

    def score(predictions, domain, split="test", threshold=None):
        assert len(predictions) == len({p["id"] for p in predictions})
        assert {p["id"] for p in predictions} == {r["id"] for r in rows if r["domain"] == domain and r["split"] == split}
        assert all(lookup[p["id"]]["label"] == p["label"] for p in predictions)
        t = predictions[0]["threshold"] if threshold is None else threshold
        assert all(np.isfinite(p["probability"]) and 0 <= p["probability"] <= 1 for p in predictions)
        return metrics([p["label"] for p in predictions], [p["probability"] for p in predictions], t)["macro_f1"]

    def labels(ids, domain, seed):
        expected = sample_budget([r for r in rows if r["domain"] == domain and r["split"] == "train"], 80, seed, domain)
        assert ids == [r["id"] for r in expected]

    sequence_root = root / args.sequence_directory
    seq = [json.loads(p.read_text()) for p in (sequence_root / "runs").glob("*.json")]
    assert len(seq) == 72
    assert {(r["seed"], r["method"], tuple(r["order"])) for r in seq} == set(itertools.product([42,43,44], ["daewc","adapter","lora","full"], itertools.permutations(domains)))
    for r in seq:
        assert len(r["stages"]) == 3 and len(r["matrix"]) == 4
        for stage in r["stages"]:
            step = stage["stage"]
            labels(stage["label_ids"], stage["domain"], r["seed"])
            assert sha256(sequence_root / stage["checkpoint"]) == stage["checkpoint_sha256"]
            pred = json.loads((sequence_root / "predictions" / f"{r['id']}_stage{step}.json").read_text())
            assert set(pred) == set(["source"] + r["order"][:step])
            for domain, records in pred.items():
                assert abs(score(records, domain) - r["matrix"][step][domain]) < 1e-9
        delta = r["matrix"][-1]["source"] - r["matrix"][0]["source"]
        differences = [r["matrix"][-1][d] - r["matrix"][j+1][d] for j,d in enumerate(r["order"][:-1])]
        assert abs(delta - r["delta_source_pp"]) < 1e-9
        assert abs(np.mean(differences) - r["bwt_older_targets_pp"]) < 1e-9
        assert abs(np.mean([delta] + differences) - r["bwt_including_source_pp"]) < 1e-9
        assert abs(np.mean([r["matrix"][-1][d] for d in domains]) - r["final_target_macro_f1"]) < 1e-9
        if r["method"] in {"adapter", "lora"}:
            assert delta == 0 and all(v == 0 for v in differences)
    mechanism = []
    if not args.skip_mechanism:
        mechanism = [json.loads(p.read_text()) for p in (root / "mechanism/runs").glob("*.json")]
        assert len(mechanism) == 81
        assert {(r["seed"],r["domain"],r["calibration_lr"],r["lambda"]) for r in mechanism} == set(itertools.product([42,43,44],domains,[1e-4,1e-3,1e-2],[0,100,10000]))
        for r in mechanism:
            labels(r["label_ids"], r["domain"], r["seed"])
            assert sha256(root / "mechanism" / r["checkpoint"]) == r["checkpoint_sha256"]
            pred = json.loads((root / "mechanism/predictions" / f"{r['id']}.json").read_text())
            assert abs(score(pred["target"], r["domain"]) - r["target"]["all"]["macro_f1"]) < 1e-9
            assert abs(score(pred["source"], "source") - r["source_after"]["all"]["macro_f1"]) < 1e-9
            assert abs(r["source_after"]["all"]["macro_f1"] - r["source_before"]["all"]["macro_f1"] - r["delta_source_pp"]) < 1e-9
    result = {"sequence_directory": args.sequence_directory, "verified_sequences": len(seq), "verified_stage_predictions": 3*len(seq),
              "verified_mechanism_runs": len(mechanism), "checks": "complete design; paired labels; test membership; probabilities; score arithmetic; checkpoint hashes; backward transfer definitions"}
    if args.confirmation:
        confirmation = [json.loads(p.read_text()) for p in (root / "confirmation/runs").glob("*.json")]
        assert len(confirmation) == 162
        plan = json.loads((root / "confirmation/frozen_plan.json").read_text())
        expected = set()
        for group, run_dir, _, config_name in plan["groups"]:
            cfg = json.loads((args.configs / config_name).read_text())
            expected |= set(itertools.product([group],cfg["methods"],domains,[42,43,44]))
        assert {(r["group"],r["method"],r["domain"],r["seed"]) for r in confirmation} == expected
        for r in confirmation:
            assert r["selection_or_training_performed"] is False and r["threshold"] == .5
            pred = json.loads((root / "confirmation/predictions" / f"{r['id']}.json").read_text())
            assert abs(score(pred,r["domain"],"dev",.5) - r["scores"]["macro_f1"]) < 1e-9
        result["verified_second_test_runs"] = len(confirmation)
    write_json(args.out,result)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    for field in ["artifacts","data","out"]:
        p.add_argument("--"+field,type=Path,required=True)
    p.add_argument("--configs",type=Path,default=Path("configs"))
    p.add_argument("--confirmation",action="store_true")
    p.add_argument("--sequence-directory",default="sequential")
    p.add_argument("--skip-mechanism",action="store_true")
    run(p.parse_args())
