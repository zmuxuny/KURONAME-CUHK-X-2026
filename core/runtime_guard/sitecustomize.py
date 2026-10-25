"""Inherited Python-child read auditing for the relocated raw-stage subprocesses."""
import atexit
import json
import os
import sys
from pathlib import Path

directory = os.environ.get('CUHKX_RUNTIME_AUDIT_DIR')
prefixes = json.loads(os.environ.get('CUHKX_FORBIDDEN_ROOTS', '[]'))
if os.environ.get('CUHKX_FORBIDDEN_PREFIX'):
    prefixes.append(os.environ['CUHKX_FORBIDDEN_PREFIX'])
observed, blocked = set(), []
sys.dont_write_bytecode = True


def audit(event, args):
    if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = str(Path(os.fsdecode(args[0])).resolve())
    mode, flags = args[1:3]
    if (mode and any(letter in mode for letter in 'wax+')) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
        return
    if any(path.startswith(prefix) for prefix in prefixes):
        blocked.append(path)
        raise PermissionError('Raw packaged stage attempted forbidden read: ' + path)
    if not path.startswith(('/usr/', '/root/miniconda3/', '/proc/', '/sys/', '/dev/')):
        observed.add(path)


def close():
    if directory:
        path = Path(directory) / (str(os.getpid()) + '.json')
        path.write_text(json.dumps({'pid': os.getpid(), 'argv': sys.argv, 'prefixes': prefixes,
                                    'blocked': blocked, 'observed_reads': sorted(observed)}, indent=2) + '\n')


if directory:
    sys.addaudithook(audit)
    atexit.register(close)
