#!/usr/bin/env python3
"""Restore the frozen public-release model files with Python's standard library."""
import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def download(url, path):
    partial = path.with_suffix(path.suffix + '.part')
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
    with urllib.request.urlopen(request, timeout=120) as response:
        append = bool(offset and response.status == 206)
        with partial.open('ab' if append else 'wb') as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
    partial.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-dir', type=Path, help='Use previously downloaded archives instead of the network')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'models.json').read_text())
    folder = args.from_dir or ROOT / '.cache/models'
    folder.mkdir(parents=True, exist_ok=True)
    for archive in manifest['archives']:
        files = archive['files']
        if all((ROOT / f['path']).is_file() and (ROOT / f['path']).stat().st_size == f['bytes'] for f in files):
            print(archive['name'], 'already installed', flush=True)
            continue
        path = folder / archive['name']
        if not path.is_file():
            if args.from_dir:
                raise FileNotFoundError(path)
            print('Downloading', archive['name'], flush=True)
            download(archive['url'], path)
        if digest(path) != archive['sha256']:
            raise ValueError(f'Archive checksum mismatch: {path}. Remove this archive and download again.')
        print('Extracting', archive['name'], flush=True)
        with tarfile.open(path, 'r:gz') as handle:
            members = handle.getmembers()
            if {m.name for m in members} != {f['path'] for f in files}:
                raise ValueError('Unexpected file set in model archive')
            handle.extractall(ROOT, members=members, filter='data')
        if not args.from_dir:
            path.unlink()  # Avoid keeping a second multi-GB copy after installation.
    print('Models ready. See MODEL_LICENSE.md for their individual terms.')


if __name__ == '__main__':
    main()
