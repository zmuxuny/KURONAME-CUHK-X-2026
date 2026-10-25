#!/usr/bin/env python3
"""Run only the frozen experts required by this explicitly grouped QA batch."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import pandas as pd

from cuhkx.acquisition_v2 import (
    EXPERT_SCOPES,
    FEATURE_ORDER,
    assemble,
    expert_acquisition,
    expert_questions,
    read_acquisition,
    read_batch,
)
from scripts import run_finalist_pipeline as base

KIND = "cuhkx_partial_acquisition_expert_inference_v2"
SOURCES = ["scripts/infer_acquisition_experts_v2.py", "src/cuhkx/acquisition_v2.py",
           "src/cuhkx/acquisition.py", "scripts/run_finalist_pipeline.py"]
ENTRYPOINTS = {"action": "scripts/hau_action_acquisition.py", "emotion": "scripts/run_emotion_recipe_nested.py",
               "sequence": "scripts/temporal_checkpoint.py", "harn": "scripts/run_harn_seed_ensemble.py"}


def checkpoint_inputs(profile, plan):
    """Trace active model assets; unused expert paths and cached features are never opened."""
    if (set(profile) != {"schema_version", "profile", "checkpoints", "features"}
            or profile["schema_version"] != 1 or profile["profile"] != "independent_equal_seed_harn"
            or not set(plan.experts) <= set(profile["checkpoints"]) <= set(EXPERT_SCOPES)
            or not set(profile["features"]) <= set(FEATURE_ORDER)):
        raise ValueError("profile must supply the selected frozen experts and recognized feature names")
    files, sources, checkpoints = {}, set(), {}

    def add(path, expected=None):
        record = base.entry(path, expected)
        files[record["path"]] = record
        return Path(record["path"])

    def collect_sources(descriptor):
        for key in ("source_hashes", "recipe_source_hashes", "original_source_hashes"):
            for path, digest in descriptor.get(key, {}).items():
                sources.add(add(base.resolve(path), digest))
        for item in descriptor.get("sources", []):
            sources.add(add(base.resolve(item["path"]), item["sha256"]))

    def model_refs(descriptor, parent):
        refs = descriptor.get("models", {})
        for item in refs.values() if isinstance(refs, dict) else refs:
            path = add(parent / item["path"], item["sha256"])
            if path.suffix == ".json":
                nested = json.loads(path.read_text())
                collect_sources(nested)
                model_refs(nested, path.parent)

    kinds = {"action": "cuhkx_full_hau_action_checkpoint_v1", "sequence": "cuhkx_repeated_pairwise_temporal_v1",
             "harn": "cuhkx_harn_equal_seed_ensemble_v1"}
    for name in plan.experts:
        path = base.resolve(profile["checkpoints"][name])
        if path.suffix != ".json":
            raise ValueError("selected expert requires a frozen JSON model descriptor")
        path = add(path)
        checkpoints[name] = path
        descriptor = json.loads(path.read_text())
        collect_sources(descriptor)
        if name == "emotion":
            if descriptor.get("model_kind") != "cuhkx_emotion_recipe_nested_v6":
                raise ValueError("partial inference requires the frozen emotion v6 policy")
        elif descriptor.get("model_kind" if name == "sequence" else "kind") != kinds[name]:
            raise ValueError(f"selected {name} model kind differs from the frozen expert")
        if name in {"action", "sequence", "emotion"}:
            suffix = ".sha256" if name == "emotion" else ".json.sha256"
            if add(path.with_suffix(suffix)).read_text().strip() != base.sha(path):
                raise ValueError("selected checkpoint descriptor sidecar differs")
        if name == "action":
            model_refs(descriptor, path.parent)
        elif name == "sequence":
            model = descriptor["model"]
            add(path.parent / model["path"], model["sha256"])
        elif name == "emotion":
            for model in (descriptor["source_manifest"], descriptor["source_model"]["checkpoint"],
                          descriptor["source_model"]["scope"]):
                add(base.resolve(model["path"]), model["sha256"])
        else:
            if descriptor.get("visual_weights") != [0.5, 0.5]:
                raise ValueError("HARn must retain the fixed equal two-seed weighting")
            for model in descriptor["classifiers"].values():
                add(base.resolve(model["path"]), model["sha256"])
    return list(files.values()), sources, checkpoints


def feature_paths(profile, plan):
    if not set(plan.features) <= set(profile["features"]):
        raise ValueError("active experts are missing required features")
    paths = {}
    for name in plan.features:
        path = base.resolve(profile["features"][name])
        if not path.is_file() or path.suffix != (".json" if name == "harn_middle" else ".npz"):
            raise ValueError(f"active feature is not the declared archive: {name}")
        paths[name] = path
    return paths


def prepare_expert_inputs(qa, acquisition, plan, inputs):
    paths, records, scopes = {}, [], {}
    for name in plan.experts:
        directory = inputs / name
        directory.mkdir()
        frame = expert_questions(qa, name)
        subset = expert_acquisition(frame, acquisition)
        item = {"qa": directory / "test_qa.csv", "scene": None, "groups": None}
        frame.to_csv(item["qa"], index=False)
        if subset is not None:
            item.update(scene=directory / "acquisition.csv", groups=directory / "groups.json")
            subset.scenes.to_csv(item["scene"], index=False)
            base.write(item["groups"], subset.temporal_manifest())
        paths[name] = item
        records.extend(base.entry(path) for path in item.values() if path is not None)
        scopes[name] = {"qa_ids": frame.qa_id.tolist(), "rows": len(frame),
                        "clips": sorted(set(frame.path)),
                        "context": "Only real questions on clips containing this expert's target type"}
    return paths, records, scopes


def stage_commands(plan, input_paths, checkpoints, features, work, workers):
    def command(script, *args):
        return [sys.executable, str(ROOT / "scripts" / script), "infer", *map(str, args)]

    commands = {}
    for name in plan.experts:
        qa_path, scene_path, groups_path = (input_paths[name][key] for key in ("qa", "scene", "groups"))
        if name != "harn" and (scene_path is None or groups_path is None):
            raise ValueError("active HAU expert commands require this batch's explicit grouping")
        output = work / name
        if name == "action":
            commands[name] = command("hau_action_acquisition.py", "--checkpoint", checkpoints[name],
                "--qa-csv", qa_path, "--logits", features["logits8"], features["logits16"],
                "--window-features", *[features[key] for key in ("windows16", "windows32", "windows48")],
                "--output-dir", output, "--workers", workers, "--scene-manifest", scene_path)
        elif name == "emotion":
            commands[name] = command("run_emotion_recipe_nested.py", "--policy", checkpoints[name],
                "--test-csv", qa_path, "--imu", features["imu"], "--skeleton", features["skeleton"],
                "--radar", features["radar"], "--visual", features["emotion_visual"],
                "--output-dir", output, "--scene-manifest", scene_path)
        elif name == "sequence":
            commands[name] = command("temporal_checkpoint.py", "--checkpoint", checkpoints[name],
                "--qa-csv", qa_path, "--logits", features["logits8"], "--logits", features["logits16"],
                "--output-dir", output, "--workers", workers, "--groups-json", groups_path)
        else:
            commands[name] = command("run_harn_seed_ensemble.py", "--checkpoint", checkpoints[name],
                "--qa-csv", qa_path, "--skeleton", features["skeleton"], "--visual", features["harn_middle"],
                "--output-dir", output)
    return commands


def run(args):
    qa, sample, plan = read_batch(args.data_dir.resolve())
    acquisition = read_acquisition(qa, args.scene_manifest)
    profile = json.loads(args.config.read_text())
    dependencies, model_sources, checkpoints = checkpoint_inputs(profile, plan)
    features = feature_paths(profile, plan)
    output = args.output.resolve()
    work, report_path = output.with_suffix(".work"), output.with_suffix(".report.json")
    decisions_path = output.with_suffix(".decisions.csv")
    if any(path.exists() for path in (output, work, report_path, decisions_path)):
        raise FileExistsError("partial expert inference requires new output paths")
    work.mkdir(parents=True)
    inputs = work / "inputs"
    inputs.mkdir()
    input_paths, generated_records, expert_scopes = prepare_expert_inputs(qa, acquisition, plan, inputs)
    commands = stage_commands(plan, input_paths, checkpoints, features, work, args.workers)
    sources = [base.entry(path) for path in base.source_closure(
        [*SOURCES, *[ENTRYPOINTS[name] for name in plan.experts], *map(str, model_sources)])]
    original_inputs = [args.config, args.data_dir / "test_qa.csv", args.data_dir / "sample_submission.csv",
                       *features.values(), *([args.scene_manifest] if acquisition is not None else [])]
    records = [*[base.entry(path) for path in original_inputs], *generated_records]
    report = {"kind": KIND, "status": "running", "batch_plan": plan.to_dict(), "inputs": records,
              "dependencies": dependencies, "sources": sources, "stages": {}, "expert_qa_scopes": expert_scopes,
              "skipped_experts": [name for name in EXPERT_SCOPES if name not in plan.experts],
              "new_fits": 0, "target_columns_used": False, "prior_predictions_read": False}
    base.write(report_path, report)
    for source in sources:
        relative = Path(source["path"]).relative_to(ROOT)
        target = work / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source["path"], target)
    frames = {}
    environment = dict(os.environ, OMP_NUM_THREADS=str(args.workers), OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS=str(args.workers))
    try:
        for name, command in commands.items():
            started = time.monotonic()
            print(f"expert={name} starting", flush=True)
            with (work / f"{name}.log").open("x") as log:
                result = subprocess.run(command, cwd=work, env=environment, stdout=log,
                                        stderr=subprocess.STDOUT, check=False)
            report["stages"][name] = {"command": command, "exit_code": result.returncode,
                                      "seconds": time.monotonic() - started}
            base.write(report_path, report)
            if result.returncode:
                raise RuntimeError(f"active expert {name} failed; see {work / (name + '.log')}")
            path = work / name / "test_predictions.csv"
            frames[name] = pd.read_csv(path, dtype=str, keep_default_na=False)
            report["stages"][name].update(rows=len(frames[name]), sha256=base.sha(path))
        submission, decisions = assemble(qa, sample, frames)
        base.verify([*records, *dependencies, *sources])
        submission.to_csv(output, index=False, lineterminator="\n")
        decisions.to_csv(decisions_path, index=False)
        report.update(status="complete", rows=len(submission), sha256=base.sha(output), format_validation="passed")
        base.write(report_path, report)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        base.write(report_path, report)
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scene-manifest", type=Path)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:args.workers])
    report = run(args)
    print(json.dumps({key: report[key] for key in ("status", "rows", "sha256", "skipped_experts")}, indent=2))


if __name__ == "__main__":
    main()
