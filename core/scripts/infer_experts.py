#!/usr/bin/env python3
"""Run all four independent experts from weights and test features, then validate CSV.

Raw feature extraction remains upstream. No stored prediction vector is an input;
only outputs created in this invocation are read for the final composition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuhkx.inference import QUESTION_COLUMNS, question_metadata
from cuhkx.submission import validate_submission_frames

SCOPES = {
    "action": {("HAU", "single"), ("HAU", "multi"), ("HAU", "combination")},
    "emotion": {("HAU", "emotion")},
    "sequence": {("HAU", "sequence")},
    "harn": {("HARn", "single"), ("HARn", "object_interaction")},
}
FEATURES = {"imu", "skeleton", "radar", "logits8", "logits16", "windows16",
            "windows32", "windows48", "harn_visual"}
SOURCES = ["scripts/infer_experts.py", "inference.sh", "src/cuhkx/inference.py",
           "src/cuhkx/submission.py"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_config(config: dict) -> tuple[dict[str, Path], dict[str, Path]]:
    if (set(config) != {"schema_version", "profile", "checkpoints", "features"}
            or config["schema_version"] != 1 or config["profile"] != "independent_local_v2"
            or set(config["checkpoints"]) != set(SCOPES) or set(config["features"]) != FEATURES):
        raise ValueError("expected the complete four-expert config; prediction inputs are unsupported")
    groups = []
    for key in ("checkpoints", "features"):
        paths = {}
        for name, raw in config[key].items():
            if not isinstance(raw, str) or not raw:
                raise ValueError(f"invalid {key} path: {name}")
            path = (ROOT / raw).resolve()
            if not path.is_file():
                raise ValueError(f"missing {key} input: {path}")
            if path.suffix != (".json" if key == "checkpoints" else ".npz"):
                raise ValueError(f"unexpected {key} input format: {path}")
            paths[name] = path
        groups.append(paths)
    return tuple(groups)


def assemble(questions: pd.DataFrame, sample: pd.DataFrame,
             frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    qa = question_metadata(questions)
    if set(frames) != set(SCOPES):
        raise ValueError("all four specialist outputs are required")
    if sample.columns.tolist() != ["qa_id", "prediction"] or sample.qa_id.tolist() != qa.qa_id.tolist():
        raise ValueError("sample schema or IDs/order differ from questions")
    scope = pd.Series(list(zip(qa.source, qa.category, strict=True)), index=qa.index)
    parts = []
    for name, valid in SCOPES.items():
        frame = frames[name]
        if not {"qa_id", "prediction"} <= set(frame) or not frame.qa_id.is_unique:
            raise ValueError(f"invalid specialist IDs/columns: {name}")
        target = qa.loc[scope.isin(valid)]
        if set(frame.qa_id) != set(target.qa_id):
            raise ValueError(f"specialist coverage differs from declared scope: {name}")
        for column in ("source", "category"):
            if column in frame:
                expected = frame.qa_id.map(target.set_index("qa_id")[column])
                if not frame[column].eq(expected).all():
                    raise ValueError(f"specialist metadata mismatch: {name}/{column}")
        parts.append(frame[["qa_id", "prediction"]].assign(expert=name))
    evidence = pd.concat(parts, ignore_index=True).set_index("qa_id")
    if not evidence.index.is_unique or set(evidence.index) != set(qa.qa_id):
        raise ValueError("specialist scopes overlap or leave missing questions")
    decisions = qa[["qa_id", "source", "category"]].copy()
    decisions["prediction"] = decisions.qa_id.map(evidence.prediction)
    decisions["expert"] = decisions.qa_id.map(evidence.expert)
    submission = decisions[["qa_id", "prediction"]].copy()
    validate_submission_frames(qa, sample, submission)
    return submission, decisions


def stage_commands(qa: Path, checkpoints: dict[str, Path], features: dict[str, Path],
                   work: Path, workers: int) -> dict[str, list[str]]:
    def command(script, *args):
        return [sys.executable, str(ROOT / "scripts" / script), "infer", *map(str, args)]

    harn_format = json.loads(checkpoints["harn"].read_text()).get("format")
    harn_scripts = {"cuhkx_harn_local_v2_checkpoint_v1": "harn_checkpoint.py",
                    "cuhkx_harn_portable_checkpoint_v2": "harn_portable_checkpoint.py"}
    if harn_format not in harn_scripts:
        raise ValueError("unknown HARn checkpoint format")
    return {
        "action": command("hau_action_checkpoint.py", "--checkpoint", checkpoints["action"],
            "--qa-csv", qa, "--logits", features["logits8"], features["logits16"],
            "--window-features", features["windows16"], features["windows32"], features["windows48"],
            "--output-dir", work / "action", "--workers", workers),
        "emotion": command("emotion_context_checkpoint.py", "--checkpoint", checkpoints["emotion"],
            "--test-csv", qa, "--imu", features["imu"], "--skeleton", features["skeleton"],
            "--radar", features["radar"], "--output-dir", work / "emotion"),
        "sequence": command("temporal_checkpoint.py", "--checkpoint", checkpoints["sequence"],
            "--qa-csv", qa, "--logits", features["logits8"], "--logits", features["logits16"],
            "--output-dir", work / "sequence", "--workers", workers),
        "harn": command(harn_scripts[harn_format], "--checkpoint", checkpoints["harn"],
            "--qa-csv", qa, "--skeleton-features", features["skeleton"],
            "--visual-features", features["harn_visual"], "--output-dir", work / "harn",
            "--workers", workers),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="contains test_qa.csv and sample_submission.csv")
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/checkpoint_inference_v2.json")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        raise ValueError("workers must be in1..3")
    output = args.output.resolve()
    work, report_path = output.with_suffix(".work"), output.with_suffix(".report.json")
    decisions_path = output.with_suffix(".decisions.csv")
    if any(path.exists() for path in (output, work, report_path, decisions_path)):
        raise ValueError("refusing to overwrite an existing inference run")
    config = json.loads(args.config.read_text())
    checkpoints, features = resolve_config(config)
    qa_path = (args.data_dir / "test_qa.csv").resolve()
    sample_path = (args.data_dir / "sample_submission.csv").resolve()
    qa = question_metadata(pd.read_csv(qa_path, usecols=QUESTION_COLUMNS,
                                       dtype=str, keep_default_na=False))
    sample = pd.read_csv(sample_path, dtype=str, keep_default_na=False)
    if sample.columns.tolist() != ["qa_id", "prediction"] or sample.qa_id.tolist() != qa.qa_id.tolist():
        raise ValueError("sample schema or IDs/order differ from questions")
    commands = stage_commands(qa_path, checkpoints, features, work, args.workers)
    work.mkdir(parents=True)
    report = {"status": "running", "profile": config["profile"], "config": config,
        "scope": "weights plus supplied test features to complete answers; raw extraction is upstream",
        "inputs": [{"path": str(path), "sha256": sha256(path)} for path in
                   [args.config.resolve(), qa_path, sample_path, *checkpoints.values(), *features.values()]],
        "source_hashes": {name: sha256(ROOT / name) for name in SOURCES}, "stages": {},
        "limits": ["Existing audited acquisition layout is required by HAU experts.",
                   "No new OOF, Private score, raw-media completeness or clean-environment claim.",
                   "K-KUNO backbone subject membership is unverified; historical validation subjects were exposed."]}

    def save_report():
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    for name in SOURCES:
        target = work / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    shutil.copyfile(args.config, work / "config.json")
    save_report()
    frames = {}
    environment = os.environ.copy()
    environment.update({key: str(args.workers) for key in
                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    try:
        for name, command in commands.items():
            started = time.perf_counter()
            print(f"expert={name} starting", flush=True)
            with (work / f"{name}.log").open("w") as handle:
                result = subprocess.run(command, cwd=work, env=environment, stdout=handle,
                                        stderr=subprocess.STDOUT, check=False)
            report["stages"][name] = {"command": command, "exit_code": result.returncode,
                "seconds": time.perf_counter() - started, "log": str(work / f"{name}.log")}
            save_report()
            if result.returncode:
                raise RuntimeError(f"{name} inference failed; see {work / (name + '.log')}")
            path = work / name / "test_predictions.csv"
            frames[name] = pd.read_csv(path, dtype=str, keep_default_na=False)
            report["stages"][name].update(rows=len(frames[name]), output_sha256=sha256(path))
            print(f"expert={name} completed rows={len(frames[name])}", flush=True)
        submission, decisions = assemble(qa, sample, frames)
        submission.to_csv(output, index=False, lineterminator="\n")
        decisions.to_csv(decisions_path, index=False, lineterminator="\n")
        report.update(status="complete", rows=len(submission), columns=submission.columns.tolist(),
                      sha256=sha256(output), output=str(output), format_validation="passed")
        save_report()
        print(json.dumps({key: report[key] for key in
                          ("status", "rows", "sha256", "output", "format_validation")}, indent=2))
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_report()
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
