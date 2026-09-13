"""Measure acquisition-cue predictability; do not treat it as veracity evidence."""
from __future__ import annotations
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .data import DATELINE, URL, canonical, clean, publisher, read_csv, sha256
from .protocol import metrics
from .run import write_json


def features(text, origin=None):
    """No topical words or content embeddings enter this diagnostic classifier."""
    n = max(1, len(text))
    value = {"log_characters": float(np.log1p(len(text))), "log_words": float(np.log1p(len(text.split()))),
             "url_count": len(URL.findall(text)), "hashtags": len(re.findall(r"(?<!\w)#\w+", text)),
             "mentions": len(re.findall(r"(?<!\w)@\w+", text)), "dateline": int(bool(DATELINE.search(text))),
             "reuters": int(bool(re.search(r"\bReuters\b", text, re.I))),
             "digit_fraction": sum(c.isdigit() for c in text)/n,
             "upper_fraction": sum(c.isupper() for c in text)/n,
             "punctuation_fraction": sum(not c.isalnum() and not c.isspace() for c in text)/n,
             "newlines": text.count("\n")}
    if origin is not None:
        value["publisher"] = origin
    return value


def cue_model():
    return make_pipeline(DictVectorizer(), StandardScaler(with_mean=False),
                         LogisticRegression(C=1., max_iter=2000, class_weight="balanced", random_state=1337))


def legacy(root, out):
    output = {"schema_version": 2, "split_seed": 1337,
              "scope": "diagnostic random holdout after exact deduplication only; never main efficacy evidence",
              "datasets": {}, "file_audit": []}
    for p in sorted(root.glob("**/*.csv")):
        if "revision_artifacts" in str(p):
            continue
        output["file_audit"].append({"path": str(p.relative_to(root)), "sha256": sha256(p), "rows": len(read_csv(p))})
    # Inspect conflicting raw full-text records separately from the headline inputs.
    a = root / "Moredata/PolitiFact_fake_news_content.csv"
    b = root / "Moredata/PolitiFact_real_news_content.csv"
    fa = {canonical(r["title"] + " " + r["text"]) for r in read_csv(a)}
    fb = {canonical(r["title"] + " " + r["text"]) for r in read_csv(b)}
    output["politifact_fulltext_conflict"] = {"fake_unique": len(fa), "real_unique": len(fb),
                                               "cross_label_identical_texts": len(fa & fb),
                                               "byte_identical_files": sha256(a) == sha256(b)}
    for name, folder, fake, real in [("legacy_isot", "news", "Fake.csv", "True.csv"),
                                    ("legacy_medical", "covid", "fakeNews.csv", "trueNews.csv")]:
        all_rows, groups = [], defaultdict(list)
        for label, filename in [(0, real), (1, fake)]:
            for row in read_csv(root / folder / filename):
                raw = row["Text"] if folder == "covid" else row["title"] + " " + row["text"]
                # Dateline can follow the title in the original article input.
                diagnostic = row["Text"] if folder == "covid" else row["text"]
                groups[clean(raw)].append({"raw": raw, "diagnostic": diagnostic, "label": label})
        conflict = 0
        for text, group in groups.items():
            if not text or len({r["label"] for r in group}) != 1:
                conflict += len(group); continue
            all_rows.append(group[0])
        train, test = train_test_split(all_rows, test_size=.2, random_state=1337,
                                       stratify=[r["label"] for r in all_rows])
        results = {}
        for version in ["raw", "scrubbed"]:
            transform = (lambda x: x) if version == "raw" else clean
            clf = cue_model()
            clf.fit([features(transform(r["diagnostic"])) for r in train], [r["label"] for r in train])
            prob = clf.predict_proba([features(transform(r["diagnostic"])) for r in test])[:, 1]
            results[version + "_format_cues"] = metrics([r["label"] for r in test], prob)
        stats = {}
        for label in [0, 1]:
            group = [r for r in all_rows if r["label"] == label]
            stats[str(label)] = {"n": len(group), "url_fraction": float(np.mean([bool(URL.search(r["diagnostic"])) for r in group])),
                                 "reuters_fraction": float(np.mean([bool(re.search(r"\bReuters\b", r["diagnostic"], re.I)) for r in group])),
                                 "hashtag_fraction": float(np.mean([bool(re.search(r"#\w+", r["diagnostic"])) for r in group])),
                                 "median_words": float(np.median([len(r["raw"].split()) for r in group]))}
        output["datasets"][name] = {"n": len(all_rows), "conflict_or_empty_rows_excluded": conflict,
                                    "train_n": len(train), "test_n": len(test), "class_stats": stats, "scores": results}
        print(name, results, flush=True)
    write_json(out / "legacy_audit.json", output)


def primary(data_root, out):
    rows = [json.loads(s) for s in (data_root / "records.jsonl").read_text().splitlines()]
    manifest = json.loads((data_root / "manifest.json").read_text())
    results = {}
    for domain in ["source", "health", "public_affairs", "entertainment"]:
        train = [r for r in rows if r["domain"] == domain and r["split"] == "train"]
        test = [r for r in rows if r["domain"] == domain and r["split"] == "test"]
        wanted = set(manifest["matched_test_ids"][domain])
        mask = [i for i, row in enumerate(test) if row["id"] in wanted]
        scores = {}
        for version in ["raw", "scrubbed", "scrubbed_with_origin"]:
            text_key = "raw_text" if version == "raw" else "text"
            def x(r):
                return features(r[text_key], r["origin"] if version == "scrubbed_with_origin" else None)
            clf = cue_model()
            clf.fit([x(r) for r in train], [r["label"] for r in train])
            p = clf.predict_proba([x(r) for r in test])[:, 1]
            scores[version] = {"all": metrics([r["label"] for r in test], p),
                               "matched": metrics([test[i]["label"] for i in mask], p[mask]) if mask else None}
        results[domain] = {"train_n": len(train), "test_n": len(test), "matched_n": len(mask), "scores": scores,
                           "training_label_budget": "full domain training split; diagnostic only, not a few-shot competitor"}
    # Reconstruct the source-only word vocabulary concern using the corrected
    # source texts. This is descriptive; the main encoder uses fixed WordPiece.
    counts = Counter(w for r in rows if r["domain"] == "source" and r["split"] == "train" for w in re.findall(r"\w+", r["text"]))
    vocabulary = {w for w, _ in counts.most_common(4998)}
    rates = {}
    for domain in results:
        tokens = [w for r in rows if r["domain"] == domain and r["split"] == "test" for w in re.findall(r"\w+", r["text"])]
        rates[domain] = {"unknown": sum(w not in vocabulary for w in tokens), "total": len(tokens),
                          "rate": sum(w not in vocabulary for w in tokens)/len(tokens)}
    write_json(out / "primary_cue_audit.json", {"domains": results, "source_word_vocabulary_oov": rates,
                                               "vocabulary_rule": "4998 most frequent source train words plus 2 reserved tokens"})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("."))
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    legacy(args.data_root, args.out)
    primary(args.prepared, args.out)
