#!/usr/bin/env python3
"""Freeze current Test inputs and a propagated read/write audit for the R34 candidate."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/20260909_r34_harn_complete_candidate_v3'

GUARD = '''import atexit,json,os,sys
from pathlib import Path
root=Path(os.environ['CUHKX_RESEARCH_INFERENCE_RUN']).resolve()
original=Path('/root/workspace/CUHK-X')
policy=json.loads((root/'guard_policy.json').read_text())
allowed=set(policy['allowed_data_reads'])
reads=[];blocked=[]
sys.dont_write_bytecode=True
def audit(event,args):
 if event=='socket.connect':
  blocked.append('network');raise PermissionError('local inference forbids network')
 if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
 path=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:3]
 writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
 if writing:
  if path.is_relative_to(original):blocked.append(str(path));raise PermissionError('original workspace is read only')
  return
 protected=path.suffix.lower() in {'.csv','.npy','.npz'}
 if path.name=='training_qa.csv' or 'training' in [p.lower() for p in path.parts] or (protected and str(path) not in allowed and not path.is_relative_to(root)):
  blocked.append(str(path));raise PermissionError('undeclared training/prediction input: '+str(path))
 if protected:reads.append(str(path))
sys.addaudithook(audit)
def finish():
 (root/'read_audit'/f'{os.getpid()}.json').write_text(json.dumps({'pid':os.getpid(),'reads':reads,'blocked':blocked,'argv':sys.argv})+'\\n')
atexit.register(finish)
'''


def main():
    config_path = ROOT / 'configs/checkpoint_inference_r34_harn_comparison_v1.json'
    config = json.loads(config_path.read_text())
    OUT.mkdir(exist_ok=False)
    shutil.copyfile(ROOT / 'runs/20260909_r34_harn_complete_candidate_v1/protection.json', OUT / 'protection.json')
    inputs = OUT / 'inputs'
    inputs.mkdir(exist_ok=False)
    old = ROOT / 'runs/20260908_videomae_harn_comparison_candidate_v1/inputs'
    for name in ['test_qa.csv', 'sample_submission.csv', 'poison_qa.csv', 'poison_restore.csv']:
        shutil.copyfile(old / name, inputs / name)
    allowed = []
    for value in config['features'].values():
        path = (ROOT / value).resolve()
        if path.suffix == '.npz':
            allowed.append(str(path))
        else:
            payload = json.loads(path.read_text())
            for name in (['features', 'metadata'] if 'metadata' in payload else ['features', 'units']):
                entry = payload[name]
                filename = entry['path'] if isinstance(entry, dict) else entry
                allowed.append(str((path.parent / filename).resolve()))
    (OUT / 'guard').mkdir()
    (OUT / 'read_audit').mkdir()
    (OUT / 'guard/sitecustomize.py').write_text(GUARD)
    (OUT / 'guard_policy.json').write_text(json.dumps({'allowed_data_reads': allowed}, indent=2) + '\n')
    paths = [config_path, Path(__file__), ROOT / 'scripts/infer_r34_harn_experts.py',
             ROOT / 'scripts/infer_harn_architecture_ensemble.py', OUT / 'guard/sitecustomize.py',
             OUT / 'guard_policy.json', OUT / 'protection.json', *inputs.iterdir()]
    entries = [{'path': str(p.resolve()), 'sha256': hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()} for p in paths]
    (OUT / 'execution_freeze.json').write_text(json.dumps({'inputs': entries, 'pseudo_dependency': False,
        'recipe': 'unchanged historical nested R18/R18/R34 .25/.25/.5, missing-family masks',
        'proposed_purpose': 'complete independent architecture branch; submit only after actual model and complete inference validation'}, indent=2) + '\n')
    print(json.dumps({'prepared': str(OUT), 'allowed_arrays': len(allowed)}))


if __name__ == '__main__':
    main()
