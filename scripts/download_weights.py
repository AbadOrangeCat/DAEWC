"""Download, verify, and restore the archived models from the matching release."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import ssl
import tempfile
import urllib.request
import zipfile


def checksum(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def verified_file(path, record):
    return path.is_file() and path.stat().st_size == record['bytes'] and checksum(path) == record['sha256']


def destination(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or '\\' in name:
        raise ValueError(f'Unsafe archive path: {name}')
    result = root.joinpath(*relative.parts).resolve()
    if not result.is_relative_to(root.resolve()) or result == root.resolve():
        raise ValueError(f'Archive path escapes the output directory: {name}')
    return result


def load_manifest(path):
    manifest = json.loads(path.read_text())
    files = {row['path']: row for row in manifest['files']}
    if len(files) != len(manifest['files']) or len(files) != manifest['file_count']:
        raise ValueError('Duplicate or missing weight records in the manifest')
    members = [name for archive in manifest['archives'] for name in archive['files']]
    if len(members) != len(set(members)) or set(members) != set(files):
        raise ValueError('Archive membership does not match the weight records')
    return manifest, files


def download(archive, directory):
    directory.mkdir(parents=True, exist_ok=True)
    target = destination(directory, archive['name'])
    if verified_file(target, archive):
        return target
    partial = target.with_suffix(target.suffix + '.part')
    # Use the installed certificate bundle where available, including Python on macOS.
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    for attempt in range(3):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset >= archive['bytes']:
                if verified_file(partial, archive):
                    partial.replace(target)
                    return target
                partial.unlink()
                offset = 0
            headers = {'User-Agent': 'DAEWC-weight-downloader'}
            if offset:
                headers['Range'] = f'bytes={offset}-'
            request = urllib.request.Request(archive['url'], headers=headers)
            with urllib.request.urlopen(request, context=context, timeout=120) as response:
                resumed = response.status == 206 and offset > 0
                if response.status == 206 and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise ValueError('Unexpected server byte range')
                with partial.open('ab' if resumed else 'wb') as output:
                    for block in iter(lambda: response.read(1024 * 1024), b''):
                        output.write(block)
            if not verified_file(partial, archive):
                partial.unlink(missing_ok=True)
                raise ValueError(f'Download checksum mismatch: {archive["name"]}')
            partial.replace(target)
            return target
        except (OSError, ValueError) as error:
            if attempt == 2:
                raise
            print(f'Retrying {archive["name"]}: {error}', flush=True)
    raise RuntimeError('Download failed')


def restore(archive_path, archive, files, root):
    with zipfile.ZipFile(archive_path) as source:
        if sorted(source.namelist()) != sorted(archive['files']):
            raise ValueError('Unexpected or duplicate ZIP members')
        # Validate every destination before creating any files.
        targets = {name: destination(root, name) for name in archive['files']}
        for name, target in targets.items():
            record = files[name]
            info = source.getinfo(name)
            if info.file_size != record['bytes']:
                raise ValueError(f'Unexpected extracted size: {name}')
            if verified_file(target, record):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temp_name = tempfile.mkstemp(prefix='.weight-', dir=target.parent)
            temporary = Path(temp_name)
            try:
                value = hashlib.sha256()
                with os.fdopen(descriptor, 'wb') as output, source.open(info) as input_file:
                    for block in iter(lambda: input_file.read(1024 * 1024), b''):
                        value.update(block)
                        output.write(block)
                if value.hexdigest() != record['sha256']:
                    raise ValueError(f'Extracted checksum mismatch: {name}')
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)


def verify_weights(root, files):
    failures = [name for name, record in files.items() if not verified_file(destination(root, name), record)]
    if failures:
        raise ValueError(f'{len(failures)} missing or changed weight files:\n' + '\n'.join(failures[:20]))
    print(f'Verified {len(files)} weight files against their recorded SHA-256 hashes.', flush=True)


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=project / 'release/WEIGHTS_MANIFEST.json')
    parser.add_argument('--root', type=Path, default=project, help='Repository root to restore')
    parser.add_argument('--download-dir', type=Path, default=project / 'weights_downloads')
    parser.add_argument('--verify-only', action='store_true', help='Check restored weights without network access')
    parser.add_argument('--local-only', action='store_true', help='Restore verified ZIP files already in --download-dir')
    args = parser.parse_args()
    manifest, files = load_manifest(args.manifest)
    if args.verify_only:
        verify_weights(args.root, files)
        return
    for index, archive in enumerate(manifest['archives'], 1):
        print(f'[{index}/{len(manifest["archives"])}] {archive["name"]}', flush=True)
        if args.local_only:
            path = destination(args.download_dir, archive['name'])
            if not verified_file(path, archive):
                raise ValueError(f'Missing or changed local archive: {archive["name"]}')
        else:
            path = download(archive, args.download_dir)
        restore(path, archive, files, args.root)
    verify_weights(args.root, files)


if __name__ == '__main__':
    main()
