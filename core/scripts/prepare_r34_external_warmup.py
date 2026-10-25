#!/usr/bin/env python3
"""Use each external backbone's own features with the unchanged target warm-up."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import torch

from cuhkx.spatial_training import write_json
from cuhkx.validation import sha256_file
from scripts.fit_r34_external_target import initialize
from scripts.prepare_r34_head_warmup import extract, warmup


def source_identity(path):
    report_path = path.with_name('report.json')
    report = json.loads(report_path.read_text())
    digest = sha256_file(path)
    if (report['model_sha256'] != digest or not report['reload_exact']
            or report['retained_precision'] != 'float32'
            or report['target_Train_or_Test_read'] is not False):
        raise ValueError('Source needs a complete source-only FP32 reload receipt')
    return {'checkpoint': str(path.resolve()), 'sha256': digest,
            'report_sha256': sha256_file(report_path), 'arm': report['arm'],
            'source_people': report['source_people'], 'source_videos': report['videos']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--raw-run', type=Path, required=True)
    parser.add_argument('--source-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['extract', 'warmup'], required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    identity = source_identity(args.source_checkpoint)
    args.output.mkdir(parents=True, exist_ok=True)
    declared = args.output / 'external_source.json'
    if declared.exists():
        if json.loads(declared.read_text()) != identity:
            raise ValueError('External feature producer changed between stages')
    else:
        write_json(declared, identity)
    snapshot = args.output / ('source_' + args.stage)
    snapshot.mkdir(exist_ok=False)
    paths = [Path(__file__), ROOT / 'scripts/fit_r34_external_target.py',
             ROOT / 'scripts/prepare_r34_head_warmup.py', ROOT / 'src/cuhkx/r34_adaptation.py',
             ROOT / 'src/cuhkx/r34_video.py', ROOT / 'src/cuhkx/harn_tail.py']
    for path in paths:
        dest = snapshot / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
    write_json(snapshot / 'manifest.json', [{'path': str(p), 'sha256': sha256_file(p)} for p in paths])
    write_json(snapshot / 'config.json', config)
    write_json(snapshot / 'invocation.json', {'argv': sys.argv, 'cwd': str(Path.cwd()),
               'python': sys.executable, 'python_version': sys.version, 'torch': torch.__version__,
               'cuda_build': torch.version.cuda, 'external_source': identity})
    subprocess.run(['git', 'archive', '--format=tar.gz',
                    '--output=' + str(snapshot.resolve() / 'committed_dependencies.tar.gz'), 'HEAD'],
                   cwd=ROOT, check=True)
    with (snapshot / 'pip_freeze.txt').open('x') as handle:
        subprocess.run([sys.executable, '-m', 'pip', 'freeze', '--all'], stdout=handle, check=True)
    producer = 'Charades-' + identity['arm'] + ' source-only'
    if args.stage == 'extract':
        source = torch.load(args.source_checkpoint, map_location='cpu', weights_only=True)
        if source['source_only'] != 'Charades official Train':
            raise ValueError('Unsupported source')
        model, _ = initialize(config, source, device='cpu')
        extract(config, args.raw_run, args.output, model=model, producer=producer)
    else:
        if json.loads((args.output / 'source_extract/config.json').read_text()) != config:
            raise ValueError('Warm-up config differs from extraction')
        warmup(config, args.output, producer=producer)


if __name__ == '__main__':
    main()
