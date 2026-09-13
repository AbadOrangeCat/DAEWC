import copy
import json
import numpy as np
import pytest
import torch
from transformers import BertConfig
from daewc.data import clean, matched_ids, record, remove_duplicates
from daewc.model import Classifier
from daewc.protocol import feasible, metrics, retention, sample_budget, source_threshold
from daewc.training import adapt, empirical_fisher, penalty, predict


def cfg():
    return dict(adapter_rank=4, domain_embedding_dim=3, adapter_dropout=0., lora_rank=2,
                ewc_lambda=10., proximity_alpha=.1, fisher_damping=1e-8,
                full_lr=.001, calibration_lr=.001, module_lr=.001, batch_size=2,
                gradient_clip=1., adapt_steps=3, lwf_temperature=2., lwf_weight=1.)


def model():
    torch.manual_seed(10)
    c = BertConfig(vocab_size=24, hidden_size=8, num_hidden_layers=2,
                   num_attention_heads=2, intermediate_size=16,
                   hidden_dropout_prob=0., attention_probs_dropout_prob=0.)
    return Classifier(config=c, pretrained=False)


def encoded():
    return {"input_ids": torch.tensor([[2, 3, 4, 0], [2, 5, 6, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]]),
            "labels": torch.tensor([0, 1]), "ids": ["source:a", "source:b"]}


def test_clean_removes_markers_without_leaving_url_fragments():
    assert clean("WASHINGTON (Reuters) - Claim 42 https://abc.example/a1 #news @bob") == "claim 42"
    assert clean("www.example.com THIS") == "this"


def test_budget_is_nested_exact_and_training_only():
    rows = [record("health", f"{y}_{i}", f"word {i}", y, "train") for y in [0,1] for i in range(8)]
    small = sample_budget(rows, 2, 42, "health")
    large = sample_budget(rows, 5, 42, "health")
    assert len(small) == 4 and {r['id'] for r in small} <= {r['id'] for r in large}
    with pytest.raises(ValueError): sample_budget(rows, 9, 42, "health")
    rows[0]["split"] = "test"
    with pytest.raises(AssertionError): sample_budget(rows, 2, 42, "health")


def test_duplicate_conflicts_and_split_priority():
    rows = [record("source", "a", "the same claim has words", 0, "train"),
            record("source", "b", "the same claim has words", 0, "test"),
            record("health", "c", "another shared claim text here", 0, "train"),
            record("health", "d", "another shared claim text here", 1, "test")]
    kept, dropped, _ = remove_duplicates(rows)
    assert [r["id"] for r in kept] == ["source:b"]
    assert len(dropped) == 3


def test_signed_retention_and_improvements_are_consistent():
    assert retention(80, 82) == 2
    assert feasible(2, 1) and not feasible(2, 1, absolute=True)
    assert not feasible(-1, 1)
    assert metrics([0,1], [.1,.9])["macro_f1"] == 100


def test_ewc_is_per_example_observed_label_gradient():
    m, data = model(), encoded()
    names = m.calibration_names()
    estimate = empirical_fisher(m, data, names, 2, 1, "cpu")
    n = sorted(names)[0]
    params = dict(m.named_parameters())
    params[n].requires_grad_(True)
    manual = torch.zeros_like(params[n])
    m.eval()
    for i in range(2):
        logits = m(data["input_ids"][i:i+1], data["attention_mask"][i:i+1])
        loss = torch.nn.functional.cross_entropy(logits, data["labels"][i:i+1])
        g = torch.autograd.grad(loss, params[n])[0]
        manual += g.square()/2
    assert torch.allclose(estimate["fisher"][n], manual, atol=1e-7)
    assert penalty(m, estimate, names, 1., .1, 1e-8).item() == 0


@pytest.mark.parametrize("method", ["daewc", "no_gate", "no_ewc", "no_proximity", "no_regularizers", "adapter", "head", "lora", "full", "full_ewc", "lwf"])
def test_training_scope_and_recorded_method(method):
    m, data = model(), encoded()
    fisher = empirical_fisher(m, data, m.shared_names(), 2, 1, "cpu")
    before = copy.deepcopy(m.state_dict())
    source_before = predict(m, data, "source", "cpu")
    m.add_domain("health", method, cfg())
    info = adapt(m, data, "health", method, cfg(), fisher, "cpu")
    names = info["trainable_names"]
    assert info["candidate_count"] == 1 and not info["source_replay"] and not info["unlabeled_target"]
    assert info["target_dev_labels_used"] == 0
    assert not any(n.startswith("heads.source") for n in names)
    if method in {"daewc", "no_gate", "no_ewc", "no_proximity", "no_regularizers"}:
        assert not any(n.startswith("encoder.embeddings") for n in names)
        assert {n for n in names if n.startswith("encoder")} == m.calibration_names()
        assert all(torch.equal(before[n], m.state_dict()[n]) for n in before if n not in names)
    if method in {"adapter", "head", "lora"}:
        assert np.allclose(source_before, predict(m, data, "source", "cpu"), atol=1e-7)
    if method == "daewc":
        assert len(m.paths["health"].adapters) == 2 and len(m.paths["health"].gates) == 2
        assert info["ewc_lambda_actual"] > 0 and info["proximity_alpha_actual"] > 0


def test_origin_length_matching_is_balanced_within_strata():
    rows = [record("x", f"{y}_{i}", "one two three four", y, "test", "publisher") for y, n in [(0,4),(1,2)] for i in range(n)]
    ids = set(matched_ids(rows))
    selected = [r for r in rows if r["id"] in ids]
    assert len(selected) == 4 and sum(r["label"] for r in selected) == 2
