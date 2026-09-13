"""Summarize the frozen second test and all prespecified paired comparisons."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from .summarize import csv_write, interval
from .run import write_json


def f1_from_counts(tp, tn, fp, fn):
    return 50 * (2*tp/(2*tp+fp+fn) + 2*tn/(2*tn+fp+fn))


def run(args):
    runs = [json.loads(p.read_text()) for p in (args.input / "runs").glob("*.json")]
    assert len(runs) == 162
    grouped = defaultdict(list)
    for r in runs:
        grouped[(r["group"],r["domain"],r["method"])].append(r)
    summaries = [{"group": g,"domain":d,"method":m,
                  **interval([r["scores"]["macro_f1"] for r in rows]),
                  "test_n":rows[0]["test_n"],"trainable_parameters":rows[0]["trainable_parameters"]}
                 for (g,d,m),rows in sorted(grouped.items())]
    domains = ["health","public_affairs","entertainment"]
    methods = ["head","full","full_ewc","adapter","lora","lwf","daewc","small_adapter","small_daewc"]
    rng = np.random.default_rng(20260913)
    paired = []
    for protocol in ["fixed","cv"]:
        predictions = {}
        observed = np.zeros((3,len(methods)))
        for domain in domains:
            labels = None
            arrays = []
            for sidx,seed in enumerate([42,43,44]):
                models = []
                for midx,method in enumerate(methods):
                    small = method.startswith("small_")
                    group = protocol + ("_low_footprint" if small else "_reference")
                    base_method = method.removeprefix("small_")
                    r = next(r for r in grouped[(group,domain,base_method)] if r["seed"] == seed)
                    pred = json.loads((args.input / "predictions" / f"{r['id']}.json").read_text())
                    identity = [(p["id"],p["label"]) for p in pred]
                    if labels is None:
                        labels = np.asarray([p["label"] for p in pred],dtype=bool)
                        reference_ids = identity
                    assert identity == reference_ids
                    models.append(np.asarray([p["probability"] >= .5 for p in pred]))
                    observed[sidx,midx] += r["scores"]["macro_f1"] / len(domains)
                arrays.append(models)
            predictions[domain] = (np.asarray(arrays),np.flatnonzero(labels),np.flatnonzero(~labels))
        samples = np.zeros((2000,len(methods)))
        for b in range(len(samples)):
            chosen_seeds = rng.integers(0,3,3)
            for arr,positive,negative in predictions.values():
                # One paired example sample is shared by every method and seed.
                pos = rng.choice(positive,len(positive),replace=True)
                neg = rng.choice(negative,len(negative),replace=True)
                selected = arr[chosen_seeds]
                tp = selected[:,:,pos].sum(axis=2)
                fp = selected[:,:,neg].sum(axis=2)
                fn = len(pos)-tp
                tn = len(neg)-fp
                samples[b] += f1_from_counts(tp,tn,fp,fn).mean(axis=0)/len(domains)
        for j,method in enumerate(methods[:-1]):
            diff = samples[:,-1]-samples[:,j]
            lo,hi = np.quantile(diff,[.025,.975])
            paired.append({"protocol":protocol,"comparison":"smaller DAEWC minus "+method,
                           "mean_difference_pp":float((observed[:,-1]-observed[:,j]).mean()),
                           "paired_bootstrap_ci95_low":float(lo),"paired_bootstrap_ci95_high":float(hi),
                           "resamples":len(samples),"seed":20260913})
    args.out.mkdir(parents=True,exist_ok=True)
    csv_write(args.out / "second_test_summary.csv",summaries)
    csv_write(args.out / "second_test_paired_bootstrap.csv",paired)
    write_json(args.out / "second_test_summary.json",{"evaluations":len(runs),"groups":summaries,"paired":paired,
               "scope":"Three fixed domains and fixed class counts; seed and item resampling, with pairing retained. No selection or further training."})
    print(f"Summarized {len(runs)} evaluations and {len(paired)} paired comparisons.")


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    run(p.parse_args())
