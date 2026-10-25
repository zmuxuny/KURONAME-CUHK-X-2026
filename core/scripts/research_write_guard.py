"""Run research code while rejecting writes through aliases into the original workspace."""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def install(original: Path):
    original = original.resolve()

    def check(value):
        if isinstance(value, (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(value)).resolve()
            if path == original or original in path.parents:
                raise PermissionError(f"research attempted original-workspace write: {path}")

    def audit(event, args):
        if event == "open":
            path, mode, flags = args
            if (mode and any(x in mode for x in "wax+")) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
                check(path)
        elif event in {"os.remove", "os.rmdir", "os.mkdir", "os.chmod", "os.truncate", "os.utime"}:
            check(args[0])
        elif event in {"os.rename", "os.link", "os.symlink"}:
            check(args[1])
            if event == "os.rename":
                check(args[0])

    sys.dont_write_bytecode = True
    sys.addaudithook(audit)


if __name__ == "__main__":
    install(Path("/root/workspace/CUHK-X"))
    sys.argv = sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name="__main__")
