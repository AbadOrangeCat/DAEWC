import copy
import pytest
from daewc.reporting import primary_source_changes, source_change_macros
from daewc.verify import verified_scores, verify_checkpoint
from daewc.data import sha256


def predictions():
    return [dict(id=str(i), label=i, probability=p, threshold=.57, origin_length_matched=True)
            for i, p in enumerate([.1, .9])]


def rows():
    return [dict(id=str(i), label=i, domain="source", split="test") for i in range(2)]


@pytest.mark.parametrize("mutation", ["one_threshold", "all_thresholds", "duplicate", "nan", "outside"])
def test_verifier_rejects_corrupt_protocol_records(mutation):
    records = predictions()
    if mutation == "one_threshold": records[1]["threshold"] = .5
    if mutation == "all_thresholds":
        for record in records: record["threshold"] = .5
    if mutation == "duplicate": records.append(copy.deepcopy(records[0]))
    if mutation == "nan": records[1]["probability"] = float("nan")
    if mutation == "outside": records[1]["probability"] = 1.01
    with pytest.raises(ValueError):
        verified_scores(records, rows(), "source", .57, {"0", "1"})


def test_verifier_scores_the_frozen_threshold():
    assert verified_scores(predictions(), rows(), "source", .57, {"0", "1"})["all"]["macro_f1"] == 100


def test_missing_checkpoint_requires_explicit_prediction_mode(tmp_path):
    checkpoint = tmp_path / "missing.pt"
    with pytest.raises(FileNotFoundError):
        verify_checkpoint(checkpoint, "unused")
    assert verify_checkpoint(checkpoint, "unused", predictions_only=True) is False


def test_prediction_mode_still_rejects_corrupt_present_checkpoint(tmp_path):
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"checkpoint fixture")
    assert verify_checkpoint(checkpoint, sha256(checkpoint), predictions_only=True) is True
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_checkpoint(checkpoint, "incorrect", predictions_only=True)


def run_records():
    return [dict(domain=domain, method="daewc", shots_per_class=80, seed=seed, delta_source_pp=value)
            for domain in ["health", "public_affairs", "entertainment"]
            for seed, value in zip([42, 43, 44], [.014, .014, .018])]


def test_prose_rounds_the_mean_not_the_individual_scores():
    records = run_records()
    assert primary_source_changes(records)["ReferenceHealthSourceChange"] == pytest.approx(.046/3)
    assert r"\newcommand{\ReferenceHealthSourceChange}{0.02}" in source_change_macros(records)


def test_prose_rejects_missing_or_duplicate_seeds():
    records = run_records()
    with pytest.raises(ValueError): primary_source_changes(records[:-1])
    with pytest.raises(ValueError): primary_source_changes(records + [records[0]])
