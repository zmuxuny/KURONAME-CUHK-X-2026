#!/usr/bin/env python3
"""Run a packaged Python stage while recording and rejecting old workspace reads."""
import json
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
sys.dont_write_bytecode = True
audit_path = Path(sys.argv[1]).resolve()
script = Path(sys.argv[2]).resolve()
forbidden = os.environ.get('CUHKX_FORBIDDEN_PREFIX')
forbidden_roots = json.loads(os.environ.get('CUHKX_FORBIDDEN_ROOTS', '[]'))
prefixes = ([forbidden] if forbidden else []) + forbidden_roots
observed, blocked = set(), []


def audit(event, args):
    if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = str(Path(os.fsdecode(args[0])).resolve())
    mode, flags = args[1:3]
    writing = (mode and any(c in mode for c in 'wax+')) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
    if not writing:
        if any(path.startswith(prefix) for prefix in prefixes):
            blocked.append(path)
            raise PermissionError('Relocated stage attempted an old workspace read: ' + path)
        if not path.startswith(('/usr/', '/root/miniconda3/', '/root/cuhkx-runtime/sat_replay_', '/proc/', '/sys/', '/dev/')):
            observed.add(path)


sys.addaudithook(audit)
sys.argv = [str(script), *sys.argv[3:]]
try:
    runpy.run_path(str(script), run_name='__main__')
finally:
    audit_path.write_text(json.dumps({'forbidden_prefix': forbidden, 'forbidden_roots': forbidden_roots, 'blocked': blocked,
                                     'observed_reads': sorted(observed)}, indent=2) + '\n')
