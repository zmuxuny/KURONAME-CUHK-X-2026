#!/usr/bin/env python3
"""Select a person-broad official Train source pack and range-read its ZIP members."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd
import requests


class RemoteZip(io.RawIOBase):
    def __init__(self, url, metadata):
        self.url = url
        self.size = int(metadata['Content-Length'])
        self.etag = metadata['ETag']
        self.position = 0
        self.session = requests.Session()

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else (self.position if whence == 1 else self.size) + offset
        return self.position

    def read(self, amount=-1):
        if amount < 0:
            amount = self.size - self.position
        amount = min(amount, self.size - self.position)
        if amount == 0:
            return b''
        if not 0 < amount < 128 * 2**20:
            raise ValueError('Unexpected unbounded ZIP range request')
        start = self.position
        end = start + amount - 1
        with self.session.get(self.url, headers={'Range': f'bytes={start}-{end}',
                               'If-Match': self.etag}, stream=True, timeout=(30, 90)) as response:
            if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{self.size}':
                raise ValueError('Server ignored range or source archive changed')
            data = response.raw.read(amount + 1)
        if len(data) != amount:
            raise ValueError('Incomplete range read')
        self.position += amount
        return data


def classes(text):
    result = set()
    for item in str(text).split(';'):
        if item:
            action, start, end = item.split()
            if float(end) > float(start):
                result.add(action)
    return result


def select(table, quota=4):
    """Source annotations only: breadth first, then inverse-frequency label coverage."""
    table = table.copy()
    table['labels'] = table.actions.map(classes)
    eligible = table.loc[table.labels.map(bool)].copy()
    frequency = Counter(c for labels in eligible.labels for c in labels)
    chosen, covered, selected_counts = [], set(), Counter()
    by_person = {person: group for person, group in eligible.groupby('subject', sort=True)}
    # Four clips per eligible person, or all available when fewer than four exist.
    for _ in range(quota):
        for group in by_person.values():
            candidates = [r for r in group.itertuples() if r.id not in chosen]
            if not candidates:
                continue
            row = min(candidates, key=lambda r: (
                -len(r.labels - covered),
                -sum(1. / ((1 + selected_counts[c]) * frequency[c]**.5) for c in r.labels),
                -float(r.quality or 0), r.id))
            chosen.append(row.id)
            covered.update(row.labels)
            selected_counts.update(row.labels)
    while covered != set(frequency):
        candidates = [r for r in eligible.itertuples() if r.id not in chosen]
        row = min(candidates, key=lambda r: (-len(r.labels - covered), -float(r.quality or 0), r.id))
        chosen.append(row.id)
        covered.update(row.labels)
    return table.set_index('id').loc[chosen].reset_index().drop(columns='labels'), eligible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    protection = json.loads((args.output / 'protection.json').read_text())
    table = pd.read_csv(args.catalog, keep_default_na=False)
    selected, eligible = select(table)
    selected.to_csv(args.output / 'selected_train.csv', index=False)
    policy = {'source_table_sha256': hashlib.sha256(args.catalog.read_bytes()).hexdigest(),
              'videos': len(selected), 'people': selected.subject.nunique(),
              'original_people': table.subject.nunique(),
              'people_without_positive_annotations': sorted(set(table.subject) - set(eligible.subject)),
              'labels': len(set().union(*(classes(x) for x in selected.actions))),
              'policy': 'Four videos per eligible Train person; source-only inverse-frequency class coverage; then add missing classes. No target error scores used.',
              'verified_policy': 'Keep No with flag: README defines verified as script matching, not temporal annotation correctness; supervision uses observed action intervals only.',
              'quality_policy': 'Reported quality breaks source-only coverage ties; missing ratings rank after rated clips without altering labels.',
              'invalid_intervals': 'Retain original endpoints; mask affected class on that video; intersect positive intervals with actual decoded media time before occupancy.',
              'selection_before_media': True, 'media_budget_bytes': 3 * 2**30,
              'budget_update_reason': 'User freed Windows D space; actual free73.6GiB; retain8GiB reserve.',
              'reserve_bytes': 8 * 2**30}
    (args.output / 'selection.json').write_text(json.dumps(policy, indent=2))
    reader = RemoteZip(protection['url'], protection['headers'])
    media = args.output / 'media'
    media.mkdir(exist_ok=False)
    receipt = []
    with zipfile.ZipFile(reader) as archive:
        members = {Path(entry.filename).stem: entry for entry in archive.infolist() if entry.filename.endswith('.mp4')}
        planned = sum(members[identity].file_size for identity in selected.id)
        if planned > policy['media_budget_bytes']:
            raise ValueError('Selected media exceed declared new source pack budget')
        disk = os.statvfs('/mnt/d')
        if disk.f_bavail * disk.f_frsize - planned < policy['reserve_bytes']:
            raise RuntimeError('Selected source pack would violate host reserve')
        (args.output / 'archive_members.json').write_text(json.dumps([
            {'id': identity, 'member': members[identity].filename,
             'bytes': members[identity].file_size, 'crc': members[identity].CRC,
             'compressed_bytes': members[identity].compress_size} for identity in selected.id], indent=2))
        print(json.dumps({**policy, 'planned_media_bytes': planned}, indent=2), flush=True)
        for index, identity in enumerate(selected.id):
            disk = os.statvfs('/mnt/d')
            if disk.f_bavail * disk.f_frsize < policy['reserve_bytes']:
                raise RuntimeError('Host reserve reached')
            data = archive.read(members[identity])  # zipfile checks each original CRC.
            path = media / (identity + '.mp4')
            path.write_bytes(data)
            receipt.append({'id': identity, 'path': str(path.absolute()), 'bytes': len(data),
                            'sha256': hashlib.sha256(data).hexdigest(), 'zip_crc_verified': True})
            (args.output / 'download_progress.json').write_text(json.dumps(receipt, indent=2))
            if (index + 1) % 20 == 0:
                print(f'Charades Train media {index + 1}/{len(selected)}', flush=True)
    (args.output / 'download_report.json').write_text(json.dumps({
        'status': 'complete', 'policy': policy, 'files': receipt,
        'total_bytes': sum(r['bytes'] for r in receipt), 'official_archive_etag': reader.etag}, indent=2))


if __name__ == '__main__':
    main()
