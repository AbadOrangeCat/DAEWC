"""Write a portable checksum inventory, excluding the inventory itself."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
args = p.parse_args()
root = args.root.resolve()
out = args.out.resolve()
rows = []
for path in sorted(root.rglob('*')):
    if not path.is_file() or path.resolve() == out:
        continue
    if any(part in {'.git', '__pycache__', '.pytest_cache', '.cache'} for part in path.relative_to(root).parts):
        continue
    if path.name == '.DS_Store':
        continue
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    rows.append({'path': str(path.relative_to(root)), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()})
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({'algorithm': 'SHA-256', 'files': rows}, indent=2))
print('Recorded', len(rows), 'files and', sum(r['bytes'] for r in rows), 'bytes.')
