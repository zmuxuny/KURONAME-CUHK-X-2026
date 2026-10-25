#!/usr/bin/env python3
"""Keep the finalist network guard while allowing local DataLoader IPC."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import run_finalist_pipeline as original
from scripts import run_finalist_pipeline_v2 as hardlinks

GUARD_SOURCE = original.GUARD_SOURCE.replace("import sys\n", "import sys\nimport socket\n").replace(
    'if event == "socket.connect":',
    'if event == "socket.connect" and args[0].family != socket.AF_UNIX:')


def stage_command(run, name, attempt):
    return [sys.executable, str(Path(__file__).resolve()), "_stage", "--output-dir", str(run),
            "--stage", name, "--stage-dir", str(attempt)]


def configure():
    hardlinks.configure()
    original.KIND = "cuhkx_raw_testing_equal_seed_finalist_v3"
    original.GUARD_SOURCE = GUARD_SOURCE
    original.stage_command = stage_command
    original.EXTRACTOR_SOURCES = [*original.EXTRACTOR_SOURCES,
                                  "scripts/run_finalist_pipeline_v3.py", "finalist_v3.sh"]


def main():
    configure()
    if len(sys.argv) > 1 and sys.argv[1] != "_stage" and "--config" not in sys.argv:
        sys.argv.extend(["--config", str(ROOT / "configs/finalist_pipeline_v3.json")])
    return original.main()


if __name__ == "__main__":
    raise SystemExit(main())
