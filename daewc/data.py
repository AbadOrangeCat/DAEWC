"""Explicit input schemas, common cleaning, and duplicate-safe split manifests."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import ssl
import unicodedata
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

import certifi
import numpy as np
from datasketch import MinHash, MinHashLSH
from sklearn.model_selection import train_test_split

LIAR_URL = "https://sites.cs.ucsb.edu/~william/data/liar_dataset.zip"
URL = re.compile(r"(?:https?://|www\.)\S+", re.I)
DATELINE = re.compile(r"^[^\n]{0,100}\(Reuters\)\s*[-–—:]?", re.I)
LABELS = {"pants-fire": 1, "false": 1, "barely-true": 1,
          "mostly-true": 0, "true": 0}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def canonical(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", html.unescape(str(text)))).strip().lower()


def clean(text):
    """Remove explicit acquisition markers; this does not remove every shortcut."""
    text = unicodedata.normalize("NFKC", html.unescape(str(text)))
    text = DATELINE.sub(" ", text)
    text = URL.sub(" ", text)
    text = re.sub(r"(?<!\w)[@#]\w+", " ", text)
    text = re.sub(r"\bReuters\b", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    # Keep numbers and ordinary words: a quantity can be part of the claim.
    return canonical(text)


def publisher(url):
    url = str(url).strip()
    if not url:
        return "unknown"
    if "://" not in url:
        url = "https://" + url
    return (urlsplit(url).hostname or "unknown").lower().removeprefix("www.")


def read_csv(path):
    csv.field_size_limit(2**28)
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def record(domain, identity, text, label, split=None, origin="unknown"):
    return {"id": f"{domain}:{identity}", "domain": domain, "raw_text": text,
            "text": clean(text), "label": int(label), "split": split,
            "origin": origin, "text_sha256": hashlib.sha256(clean(text).encode()).hexdigest()}


def load_inputs(root, cache):
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "liar_dataset.zip"
    if not archive.exists():
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(LIAR_URL, context=context, timeout=90) as response:
            archive.write_bytes(response.read())
    manifests = [{"path": str(archive), "url": LIAR_URL, "sha256": sha256(archive)}]
    rows, excluded = [], Counter()
    with zipfile.ZipFile(archive) as z:
        for filename, split in [("train.tsv", "train"), ("valid.tsv", "dev"), ("test.tsv", "test")]:
            for values in csv.reader(z.read(filename).decode().splitlines(), delimiter="\t", quoting=csv.QUOTE_NONE):
                if len(values) != 14:
                    raise ValueError(f"Unexpected LIAR schema in {filename}: {len(values)} fields")
                if values[1] not in LABELS:
                    excluded["liar_half_true"] += 1
                    continue
                domain = "health" if "health-care" in values[3].split(",") else "source"
                row = record(domain, values[0], values[2], LABELS[values[1]], split, "politifact-statements")
                row["original_label"] = values[1]
                row["subjects"] = values[3]
                rows.append(row)
    for domain, prefix in [("public_affairs", "politifact"), ("entertainment", "gossipcop")]:
        for label, name in [(0, "real"), (1, "fake")]:
            path = root / "Moredata" / f"{prefix}_{name}.csv"
            manifests.append({"path": str(path.relative_to(root)), "sha256": sha256(path),
                              "url": "https://github.com/KaiDMML/FakeNewsNet"})
            for row in read_csv(path):
                if not row["title"].strip():
                    excluded[f"{domain}_empty_title"] += 1
                    continue
                rows.append(record(domain, row["id"], row["title"], label,
                                   origin=publisher(row["news_url"])))
    return rows, manifests, excluded


def shingles(text):
    words = re.findall(r"\w+", text)
    if len(words) < 3:
        return {text}
    return {" ".join(words[i:i+3]) for i in range(len(words)-2)}


def remove_duplicates(rows):
    """One representative per verified cluster, including across datasets.

    MinHash finds candidates; actual word-trigram Jaccard >= .85 joins them.
    Conflicting-label clusters are excluded. Official test/dev membership has
    priority over train when choosing a representative; no scores are examined.
    The candidate search is approximate, so no exhaustive paraphrase claim is made.
    """
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def join(i, j):
        parent[find(i)] = find(j)
    lookup, shingle_sets = {}, []
    lsh = MinHashLSH(threshold=0.70, num_perm=64)
    near_edges = exact_edges = 0
    for i, row in enumerate(rows):
        text = row["text"]
        grams = shingles(text)
        shingle_sets.append(grams)
        if text in lookup:
            join(i, lookup[text]); exact_edges += 1
            continue
        mh = MinHash(num_perm=64, seed=1337)
        mh.update_batch([s.encode() for s in sorted(grams)])
        for key in lsh.query(mh):
            j = int(key)
            other = shingle_sets[j]
            if len(grams & other) / len(grams | other) >= .85:
                join(i, j); near_edges += 1
        lsh.insert(str(i), mh)
        lookup[text] = i
        if i and i % 10000 == 0:
            print(f"Duplicate audit: {i}/{len(rows)}", flush=True)
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)
    kept, dropped = [], []
    priority = {"test": 0, "dev": 1, "train": 2, None: 3}
    for members in groups.values():
        labels = {rows[i]["label"] for i in members}
        if len(labels) > 1:
            dropped.extend({"id": rows[i]["id"], "reason": "conflicting_label_cluster"} for i in members)
            continue
        chosen = min(members, key=lambda i: (priority[rows[i]["split"]], rows[i]["id"]))
        row = rows[chosen]
        if not row["text"]:
            dropped.append({"id": row["id"], "reason": "empty_after_cleaning"})
            continue
        row["cluster"] = min(rows[i]["id"] for i in members)
        kept.append(row)
        dropped.extend({"id": rows[i]["id"], "reason": "duplicate_cluster"} for i in members if i != chosen)
    return kept, dropped, {"exact_edges": exact_edges, "verified_near_edges": near_edges,
                           "candidate_num_perm": 64, "candidate_threshold": .70,
                           "verified_jaccard": .85}


def matched_ids(rows, seed=1337):
    """Balance classes within origin and fixed word-count bins, for evaluation.

    Defined before fitting models. Never used for training, selection or tuning.
    Missing origins are excluded from this origin-controlled analysis.
    """
    bins = [0, 8, 16, 32, 64, 128, float("inf")]
    groups = defaultdict(lambda: {0: [], 1: []})
    for row in rows:
        if row["origin"] == "unknown":
            continue
        key = (row["origin"], int(np.digitize(len(row["text"].split()), bins)))
        groups[key][row["label"]].append(row["id"])
    rng = np.random.default_rng(seed)
    result = []
    for key, classes in sorted(groups.items()):
        n = min(len(classes[0]), len(classes[1]))
        for label in [0, 1]:
            ids = sorted(classes[label])
            result.extend(rng.choice(ids, n, replace=False).tolist())
    return sorted(result)


def prepare(root, out):
    out.mkdir(parents=True, exist_ok=True)
    rows, manifests, excluded = load_inputs(root, out / "downloads")
    raw_counts = dict(Counter(row["domain"] for row in rows))
    rows, dropped, duplicate_settings = remove_duplicates(rows)
    for domain in ["public_affairs", "entertainment"]:
        domain_rows = sorted([r for r in rows if r["domain"] == domain], key=lambda r: r["id"])
        train_dev, test = train_test_split(domain_rows, test_size=.2, random_state=1337,
                                          stratify=[r["label"] for r in domain_rows])
        train, dev = train_test_split(train_dev, test_size=.125, random_state=1337,
                                     stratify=[r["label"] for r in train_dev])
        for split, subset in [("train", train), ("dev", dev), ("test", test)]:
            for row in subset:
                row["split"] = split
    counts, matched = {}, {}
    for domain in sorted({r["domain"] for r in rows}):
        counts[domain] = {}
        for split in ["train", "dev", "test"]:
            subset = [r for r in rows if r["domain"] == domain and r["split"] == split]
            counts[domain][split] = {str(k): v for k, v in sorted(Counter(r["label"] for r in subset).items())}
        matched[domain] = matched_ids([r for r in rows if r["domain"] == domain and r["split"] == "test"])
    with (out / "records.jsonl").open("w") as f:
        for row in sorted(rows, key=lambda r: r["id"]):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    info = {"schema_version": 2, "inputs": manifests, "raw_counts": raw_counts,
            "counts": counts, "excluded_initial": dict(excluded), "duplicate_settings": duplicate_settings,
            "removed_records": len(dropped), "records_sha256": sha256(out / "records.jsonl"),
            "target_dev_labels_used": 0, "target_threshold": .5,
            "source_definition": "LIAR except health-care; half-true excluded",
            "health_definition": "LIAR subject contains health-care; half-true excluded",
            "matched_test_ids": matched}
    (out / "manifest.json").write_text(json.dumps(info, indent=2))
    (out / "excluded.json").write_text(json.dumps(dropped, indent=2))
    print(json.dumps({"counts": counts, "removed": len(dropped),
                      "matched_test_sizes": {k: len(v) for k, v in matched.items()}}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("."))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    prepare(args.data_root.resolve(), args.out.resolve())
