#!/usr/bin/env python3
"""Evaluate local OOF predictions using official labels and subject-level uncertainty.

Default scope is every training QA. Repeat --scope SOURCE:CATEGORY to evaluate an
explicit union of complete slices. ID-aligned OOF rows may arrive in fold order;
--require-input-order additionally enforces the official QA order. Only qa_id and
prediction are read from prediction files, never their answer/subject/mask columns.
Relative provenance paths in a protocol resolve relative to that protocol file.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuhkx.validation import (
    QA_COLUMNS,
    attach_predictions,
    evaluation_report,
    prediction_path,
    prepare_qa,
    sha256_file,
    validate_protocol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--scope", action="append", default=[])
    parser.add_argument("--require-input-order", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260905)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {args.output_dir}")
    inputs = []

    def register(path: Path, role: str, columns: list[str] | None = None) -> Path:
        path = path.resolve()
        inputs.append({"path": str(path), "role": role, "sha256": sha256_file(path),
                       "size_bytes": path.stat().st_size, "parsed_columns": columns})
        return path

    qa_path = register(args.train_csv, "official training metadata and labels", QA_COLUMNS)
    qa = pd.read_csv(qa_path, usecols=QA_COLUMNS, dtype=str, keep_default_na=False)
    all_qa = prepare_qa(qa)
    frame = prepare_qa(qa, args.scope)
    order_checks = {}
    for column, path in (("prediction", args.predictions),
                         ("baseline_prediction", args.baseline_predictions)):
        if path is None:
            continue
        path = register(prediction_path(path), column, ["qa_id", "prediction"])
        predictions = pd.read_csv(
            path, usecols=["qa_id", "prediction"], dtype=str, keep_default_na=False
        )
        frame, order_checks[column] = attach_predictions(
            frame, predictions, column=column, require_order=args.require_input_order
        )
    protocol_path = register(args.protocol, "declared validation protocol")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    audit = validate_protocol(
        protocol, frame, set(all_qa["subject"]), base_dir=protocol_path.parent
    )
    validation = evaluation_report(
        frame, samples=args.bootstrap_samples, seed=args.bootstrap_seed, confidence=args.confidence
    )
    for record in inputs:
        if sha256_file(Path(record["path"])) != record["sha256"]:
            raise ValueError(f"evaluation input changed during execution: {record['path']}")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    report = {
        "schema_version": 1, "created_utc": datetime.now(UTC).isoformat(),
        "scope": args.scope or "all official training QA rows",
        "input_order_matches_official_scope": order_checks,
        "output_order": "official training QA order within declared scope",
        "protocol_audit": audit, "validation": validation, "inputs": inputs,
        "git_commit": commit,
        "source_code": [
            {"path": str(path), "sha256": sha256_file(path)} for path in (
                Path(__file__).resolve(), ROOT / "src/cuhkx/validation.py",
                ROOT / "src/cuhkx/metadata.py", ROOT / "src/cuhkx/submission.py",
            )
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    frame["correct"] = frame["prediction"].eq(frame["answer"])
    if "baseline_prediction" in frame:
        frame["baseline_correct"] = frame["baseline_prediction"].eq(frame["answer"])
        frame["changed_from_baseline"] = frame["prediction"].ne(frame["baseline_prediction"])
    output = args.output_dir / "evaluated_rows.csv"
    frame.to_csv(output, index=False, lineterminator="\n")
    report["output"] = {"path": str(output.resolve()), "sha256": sha256_file(output)}
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    report = evaluate(parse_args())
    print(json.dumps({
        "protocol_audit": report["protocol_audit"],
        "overall": report["validation"]["overall"],
        "subject_macro_accuracy": report["validation"]["subject_macro_accuracy"],
        "worst_subject_accuracy": report["validation"]["worst_subject_accuracy"],
        "cluster_bootstrap": report["validation"]["cluster_bootstrap"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
