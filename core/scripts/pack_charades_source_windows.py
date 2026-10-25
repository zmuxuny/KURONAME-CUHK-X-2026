#!/usr/bin/env python3
"""Pack exact physical frames once instead of repeatedly decompressing windows."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import pandas as pd
import torch

from cuhkx.charades_packed_source import pack_video
from cuhkx.validation import sha256_file


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    ids = pd.read_csv(args.selection).id.tolist()
    assert len(ids) == len(set(ids)) == 2000
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / 'plan.json', {'selection_sha256': sha256_file(args.selection),
        'script_sha256': sha256_file(Path(__file__)), 'module_sha256': sha256_file(ROOT / 'src/cuhkx/charades_packed_source.py'),
        'budget_bytes': 14 * 2**30, 'server_reserve_bytes': 20 * 2**30,
        'purpose': 'Performance cache, same pixels/labels/masks/order, no training or new augmentation.'})
    total = 0
    completed = {'rgb': 0, 'depth': 0}
    try:
        for arm in ['rgb', 'depth']:
            output = args.output / arm
            output.mkdir()
            receipts = {}
            for identity in ids:
                receipt_path = args.inputs / ('receipts_' + arm) / (identity + '.json')
                while not receipt_path.exists():
                    state = json.loads((args.inputs / (arm + '_status.json')).read_text())
                    if state['status'] in ['failed', 'complete2000']:
                        raise RuntimeError('Producer terminal but required input is missing')
                    time.sleep(10)
                source = args.inputs / arm / (identity + '.npz')
                source_receipt = json.loads(receipt_path.read_text())
                assert sha256_file(source) == source_receipt['sha256']
                # An expanded-video source is an upper bound on its packed frame storage.
                import zipfile
                with zipfile.ZipFile(source) as archive:
                    upper = sum(x.file_size for x in archive.infolist()) + 4096
                if total + upper > 14 * 2**30 or shutil.disk_usage(output).free - upper < 20 * 2**30:
                    raise RuntimeError('Declared cache budget or server reserve reached')
                record = pack_video(source, output / identity)
                write(output / identity / 'receipt.json', record)
                total += sum(v['bytes'] for v in record['files'].values())
                receipts[identity] = sha256_file(output / identity / 'receipt.json')
                completed[arm] += 1
                write(args.output / 'status.json', {'status': 'packing', 'completed': completed, 'cache_bytes': total})
                if completed[arm] % 50 == 0:
                    print(arm, completed[arm], '/', len(ids), 'bytes', total, flush=True)
            write(output / 'report.json', {'status': 'complete2000', 'videos': ids, 'receipts_sha256': receipts})
        write(args.output / 'status.json', {'status': 'complete2000_both_arms', 'completed': completed, 'cache_bytes': total})
    except Exception as error:
        write(args.output / 'status.json', {'status': 'failed_preserved', 'completed': completed,
              'cache_bytes': total, 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
