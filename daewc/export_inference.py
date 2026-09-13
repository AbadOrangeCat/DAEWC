"""Precompute domain gates without changing the trained prediction function.

Each gate depends only on the known domain and saved parameters. Its projection
and domain vector can therefore be replaced by the resulting feature scales.
These files are for inference; the original training checkpoints remain intact.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .data import sha256
from .model import Classifier, tokenizer_from
from .run import write_json
from .training import encode, predict


class PrecomputedDomainPath(nn.Module):
    def __init__(self, path):
        super().__init__()
        if not path.use_gate:
            raise ValueError("This export requires a gated DAEWC path")
        self.adapters = path.adapters
        with torch.no_grad():
            scales = torch.stack([2*torch.sigmoid(g(path.embedding)) for g in path.gates])
        self.register_buffer("gate_scales",scales)

    def forward(self, x, layer):
        return self.adapters[layer](x) * self.gate_scales[layer]


def fold_domain(model, domain):
    model.paths[domain] = PrecomputedDomainPath(model.paths[domain])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_export(model_path, source_checkpoint, export_checkpoint, cfg):
    """Reconstruct a complete inference model from source and exported update."""
    payload = torch.load(export_checkpoint,map_location="cpu",weights_only=True)
    if sha256(source_checkpoint) != payload["source_checkpoint_sha256"]:
        raise ValueError("Export requires a different source checkpoint")
    model = Classifier(model_path,pretrained=False)
    model.load_state_dict(torch.load(source_checkpoint,map_location="cpu",weights_only=True))
    model.add_domain(payload["domain"],"daewc",cfg)
    fold_domain(model,payload["domain"])
    status = model.load_state_dict(payload["parameters"],strict=False)
    assert not status.unexpected_keys
    return model


def run(args):
    torch.set_num_threads(2)
    cfg=json.loads(args.config.read_text())
    rows=[json.loads(s) for s in (args.data / "records.jsonl").read_text().splitlines()]
    runs=[json.loads(p.read_text()) for p in (args.runs / "runs").glob("*.json")]
    runs=[r for r in runs if r["method"] == "daewc"]
    assert len(runs) == len(cfg["shots"])*len(cfg["domains"])*len(cfg["seeds"])
    tokenizer=tokenizer_from(args.model_path)
    data={d:encode(tokenizer,[r for r in rows if r["domain"]==d and r["split"]=="test"],cfg["max_length"]) for d in cfg["domains"]}
    args.out.mkdir(parents=True,exist_ok=True)
    summary=[]
    for seed in cfg["seeds"]:
        source_path=args.runs / f"source_seed{seed}/model.pt"
        source=Classifier(args.model_path,pretrained=False)
        source.load_state_dict(torch.load(source_path,map_location="cpu",weights_only=True))
        for r in [r for r in runs if r["seed"]==seed]:
            original=copy.deepcopy(source)
            original.add_domain(r["domain"],"daewc",cfg)
            original.load_state_dict(torch.load(args.runs / r["checkpoint"],map_location="cpu",weights_only=True)["parameters"],strict=False)
            original.eval()
            expected=predict(original,data[r["domain"]],r["domain"],"cpu",64)
            transformed=fold_domain(copy.deepcopy(original),r["domain"])
            all_state=transformed.state_dict()
            wanted=set(r["training"]["trainable_names"]) | {f"paths.{r['domain']}.gate_scales"}
            parameters={k:v for k,v in all_state.items() if k in wanted}
            dest=args.out / f"{r['run_id']}.pt"
            payload={"parameters":parameters,"domain":r["domain"],"source_seed":seed,
                     "source_checkpoint_sha256":sha256(source_path),"training_checkpoint_sha256":r["checkpoint_sha256"],
                     "inference_only":True,"gate_precomputed":True}
            torch.save(payload,dest)
            # Verify the serialized reconstruction, not just the in-memory copy.
            restored=load_export(args.model_path,source_path,dest,cfg)
            actual=predict(restored,data[r["domain"]],r["domain"],"cpu",64)
            error=float(np.max(np.abs(np.asarray(expected)-np.asarray(actual))))
            assert error == 0, (r["run_id"],error)
            total=sum(v.numel() for v in parameters.values())
            domain=sum(v.numel() for k,v in parameters.items() if k.startswith(("paths.","heads.")))
            summary.append({"run_id":r["run_id"],"export":dest.name,"sha256":sha256(dest),
                            "stored_update_values":total,"domain_specific_values":domain,
                            "parameter_payload_bytes_fp32":4*total,"domain_payload_bytes_fp32":4*domain,
                            "max_probability_error_after_serialization":error,"evaluated_examples":len(actual)})
    write_json(args.out / "export_manifest.json",{"runs":summary,"verified_exports":len(summary),
               "device":"CPU; float32; identical batches before and after export",
               "source_state":"One shared source-derived encoder; exported updates include the changed calibration values",
               "storage_scope":"Tensor elements at four bytes each; serialization headers are additional",
               "purpose":"Inference only; no training, tuning, or score-based selection"})
    print(f"Verified {len(summary)} serialized exports with identical probabilities.",flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    for field in ["config","data","runs","out"]:
        p.add_argument("--"+field,type=Path,required=True)
    p.add_argument("--model-path",required=True)
    run(p.parse_args())
