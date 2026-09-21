"""Check the bytes of all files listed in the repository release manifest."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "MANIFEST_SHA256.json").read_text())
    failures = []
    seen = set()
    for record in manifest["files"]:
        name = record["path"]
        path = (root / name).resolve()
        if name in seen or not path.is_relative_to(root) or not path.is_file():
            failures.append(name)
            continue
        seen.add(name)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if path.stat().st_size != record["bytes"] or digest.hexdigest() != record["sha256"]:
            failures.append(name)
    if failures:
        raise SystemExit("Missing, duplicate, unsafe, or changed files:\n" + "\n".join(failures))
    print(f"Verified SHA-256 and byte counts for {len(seen)} release files.")


if __name__ == "__main__":
    main()
