import hashlib
import json
import zipfile
import pytest
from scripts.download_weights import destination, load_manifest, restore, verify_weights


def fixture_archive(tmp_path, name='models/example.pt', payload=b'weight test fixture'):
    path = tmp_path / 'weights.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr(name, payload)
    record = dict(path=name, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    return path, {'files': [name]}, {name: record}


def test_restored_weights_match_exact_bytes_and_detect_later_corruption(tmp_path):
    path, archive, files = fixture_archive(tmp_path)
    root = tmp_path / 'repository'
    restore(path, archive, files, root)
    verify_weights(root, files)
    (root / 'models/example.pt').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='missing or changed'):
        verify_weights(root, files)


@pytest.mark.parametrize('name', ['../outside.pt', '/absolute.pt', r'models\outside.pt'])
def test_rejects_unsafe_extraction_paths(tmp_path, name):
    with pytest.raises(ValueError):
        destination(tmp_path, name)


def test_corrupt_archive_cannot_overwrite_an_existing_checkpoint(tmp_path):
    path, archive, files = fixture_archive(tmp_path)
    root = tmp_path / 'repository'
    target = root / 'models/example.pt'
    target.parent.mkdir(parents=True)
    target.write_bytes(b'original checkpoint')
    files['models/example.pt']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='checksum mismatch'):
        restore(path, archive, files, root)
    assert target.read_bytes() == b'original checkpoint'
    assert list(target.parent.glob('.weight-*')) == []


def test_rejects_unlisted_zip_members(tmp_path):
    path, archive, files = fixture_archive(tmp_path)
    with zipfile.ZipFile(path, 'a') as output:
        output.writestr('extra.pt', b'unlisted')
    with pytest.raises(ValueError, match='Unexpected or duplicate'):
        restore(path, archive, files, tmp_path / 'repository')


def test_rejects_manifest_with_duplicated_archive_members(tmp_path):
    _, archive, files = fixture_archive(tmp_path)
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps({'files': list(files.values()), 'file_count': 1,
                                'archives': [archive, archive]}))
    with pytest.raises(ValueError, match='Archive membership'):
        load_manifest(path)
