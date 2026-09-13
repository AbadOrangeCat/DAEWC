"""Download only the pinned model files needed by the revision."""
import argparse
import json
from pathlib import Path
from huggingface_hub import snapshot_download

project = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, default=project / 'models')
args = p.parse_args()
for config_name, folder in [('local.json', 'bert-tiny'), ('bert_base.json', 'bert-base')]:
    cfg = json.loads((project / 'configs' / config_name).read_text())
    snapshot_download(repo_id=cfg['model_id'], revision=cfg['model_revision'],
                      local_dir=args.out / folder,
                      allow_patterns=['config.json', 'vocab.txt', 'model.safetensors'])
    print('Downloaded pinned files for', cfg['model_id'], 'to', args.out / folder)
