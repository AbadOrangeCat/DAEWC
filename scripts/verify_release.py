"""Check that the delivered experiment design is complete and internally paired."""
import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

project = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project))
from daewc.data import sha256
from daewc.run import code_hash

p = argparse.ArgumentParser()
p.add_argument('--artifacts', type=Path, default=project / 'revision_artifacts')
p.add_argument('--out', type=Path)
args = p.parse_args()
root = args.artifacts
load = lambda path: json.loads(path.read_text())
expected_counts = {'local': 288, 'budget_cv': 126, 'random': 45,
                   'bert_base': 63, 'low_footprint': 108, 'low_footprint_cv': 36}
all_runs = {}
paired = {}
current_hash = code_hash()
data_hash = sha256(root / 'data/records.jsonl')
for group, count in expected_counts.items():
    folder = root / group
    records = [load(path) for path in (folder / 'runs').glob('*.json')]
    assert len(records) == count, (group, len(records), count)
    plan = load(folder / 'frozen_plan.json')
    assert plan.get('core_code_hash', plan.get('code_sha256')) == current_hash
    assert plan.get('data_hash', plan.get('data_sha256')) == data_hash
    if 'base_configuration' in plan:
        cfg = plan['base_configuration']
        wanted = set(itertools.product(cfg['domains'], plan['shots'], cfg['seeds'], plan['methods']))
        assert plan['cv_code_hash'] == sha256(project / 'daewc/budget_cv.py')
    else:
        cfg = plan['configuration']
        wanted = set(itertools.product(cfg['domains'], cfg['shots'], cfg['seeds'], cfg['methods']))
        wanted.update(itertools.product(cfg['domains'], cfg.get('ablation_shots', []),
                                        cfg['seeds'], cfg.get('ablation_methods', [])))
    observed = {(r['domain'], r['shots_per_class'], r['seed'], r['method']) for r in records}
    assert wanted == observed and len(observed) == len(records), group
    for r in records:
        key = (r['domain'], r['shots_per_class'], r['seed'])
        identifiers = tuple(sorted(r['target_label_ids']))
        assert paired.setdefault(key, identifiers) == identifiers, (group, r['run_id'])
    all_runs[group] = records

for domain, _, seed in paired:
    budgets = sorted(k for d, k, s in paired if (d, s) == (domain, seed))
    for smaller, larger in zip(budgets, budgets[1:]):
        assert set(paired[(domain, smaller, seed)]) < set(paired[(domain, larger, seed)])

provenance = load(root / 'low_footprint/source_import.json')
for seed, hashes in provenance['checkpoints'].items():
    for filename, digest in hashes.items():
        assert sha256(root / 'local' / f'source_seed{seed}' / filename) == digest
        assert sha256(root / 'low_footprint' / f'source_seed{seed}' / filename) == digest

small80 = [r for r in all_runs['low_footprint'] if r['method'] == 'daewc' and r['shots_per_class'] == 80]
assert len(small80) == 9
assert all(abs(r['delta_source_pp']) < 1 for r in small80)
assert {r['training']['trainable_parameters'] for r in small80} == {6154}
assert {r['training']['trainable_parameters'] for r in all_runs['local'] if r['method'] == 'lora'} == {8450}

exports = load(root / 'inference_exports/export_manifest.json')
assert exports['verified_exports'] == len(exports['runs']) == 36
for r in exports['runs']:
    assert r['stored_update_values'] == 5126 and r['domain_specific_values'] == 1798
    assert r['max_probability_error_after_serialization'] == 0
    assert sha256(root / 'inference_exports' / r['export']) == r['sha256']
for group in ['stability', 'stability_low']:
    info = load(root / group / 'stability_diagnostics.json')
    assert info['runs'] == info['objective_condition_satisfied'] == info['drift_bound_satisfied'] == 36
    assert info['source_gradient_bound_certified'] is False

comparisons = list(csv.DictReader((root / 'tables/confirmation/second_test_paired_bootstrap.csv').open()))
assert len(comparisons) == 16
for group in ['sequential', 'sequential_low']:
    plan = load(root / group / 'frozen_plan.json')
    assert plan['core_code_hash'] == current_hash
    assert plan['sequential_code_hash'] == sha256(project / 'daewc/sequential.py')
    assert plan['data_hash'] == data_hash
assert load(root / 'mechanism/frozen_plan.json')['code_sha256'] == sha256(project / 'daewc/mechanism_sweep.py')

report = {'status': 'passed', 'main_and_selection_final_models': expected_counts,
          'core_code_sha256': current_hash, 'data_sha256': data_hash,
          'cross_protocol_paired_budget_groups': len(paired),
          'source_checkpoint_and_fisher_imports_verified': 6,
          'smaller_k80_absolute_source_change_max_pp': max(abs(r['delta_source_pp']) for r in small80),
          'exact_serialized_exports': 36, 'objective_conditions_verified': 72,
          'complete_second_test_comparisons': 16,
          'scope': 'Design completeness, shared samples, frozen code, exact source reuse, and headline claim checks. Detailed metric/checkpoint checks are in the accompanying verification reports.'}
out = args.out or root / 'verification_release.json'
out.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
