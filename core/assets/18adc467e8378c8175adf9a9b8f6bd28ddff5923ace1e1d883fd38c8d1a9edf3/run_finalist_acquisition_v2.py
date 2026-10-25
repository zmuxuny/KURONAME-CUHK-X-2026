#!/usr/bin/env python3
"""Run selected frozen finalist stages for any supported nonempty QA subset."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "src")]

import numpy as np

from cuhkx.acquisition_v2 import STAGE_ORDER, raw_modalities, read_acquisition, read_batch
from cuhkx.submission import validate_submission_files
from scripts import infer_acquisition_experts_v2 as experts
from scripts import run_finalist_pipeline as base
from scripts import run_finalist_pipeline_v3 as ipc

KIND = "cuhkx_raw_partial_acquisition_finalist_v2"
BASE_EXECUTE_STAGE = base.execute_stage
BASE_EXTRACTOR_SOURCES = tuple(base.EXTRACTOR_SOURCES)
SOURCES = ["scripts/run_finalist_acquisition_v2.py", "scripts/infer_acquisition_experts_v2.py",
           "src/cuhkx/acquisition_v2.py", "src/cuhkx/acquisition.py", "delivery/inference_v2.sh"]


def collect_raw(raw_root, config, plan):
    roots = {key: (raw_root / config["layout"][key]).resolve() for key in ("visual", "sensors")}
    records, modalities = [], []
    for unit in plan.units:
        required, optional, sensors = raw_modalities(plan, unit)
        for modality in (*required, *optional):
            path = roots["visual"] / unit / modality / f"{modality}.mp4"
            if not path.is_file():
                if modality in required:
                    raise FileNotFoundError(f"active expert requires raw {modality}: {path}")
                continue
            records.append({"kind": "visual", "relative": str(path.relative_to(roots["visual"])),
                            **base.raw_entry(path)})
        present = []
        for modality in sensors:
            files = sorted(path for path in (roots["sensors"] / unit / modality).rglob("*") if path.is_file())
            if files:
                present.append(modality)
            for path in files:
                records.append({"kind": "sensors", "relative": str(path.relative_to(roots["sensors"])),
                                **base.raw_entry(path)})
        modalities.append({"unit": unit, "modalities": ";".join(present)})
    return records, modalities


def check_sensor_coverage(plan, modalities):
    available = {row["unit"]: row["modalities"].split(";") for row in modalities}
    for name in plan.features:
        if name not in base.SENSORS:
            continue
        modality = {"imu": "IMU", "radar": "Radar"}.get(name, "Skeleton")
        if not any(modality in available[unit] for unit in plan.feature_units[name]):
            raise ValueError(f"active {name} stage has no raw {modality} data; frozen extractor needs one valid recording")


def extraction_dependencies(config, plan):
    records = []
    needed_kuno = ["yolo11n.pt"]
    if {"logits8", "logits16"} & set(plan.features):
        needed_kuno += ["ensemble_packed.pt", "ig65m_models.py"]
    for name in needed_kuno:
        records.append(base.entry(base.resolve(config["kuno_model_dir"]) / name, config["kuno_files"][name]))
    if {"emotion_visual", "harn_middle"} & set(plan.features):
        records.append(base.entry(base.resolve(config["generic_weights"]), config["generic_weights_sha256"]))
    if "emotion_visual" in plan.features:
        records.append(base.entry(base.resolve(config["generic_config"])))
    records.append(base.entry(base.resolve(config["dependencies_lock"])))
    return records


def configure(plan):
    ipc.configure()
    base.KIND, base.FEATURES, base.STAGES = KIND, plan.features, plan.stages
    base.stage_command, base.execute_stage, base.feature_index = stage_command, execute_stage, feature_index
    base.EXTRACTOR_SOURCES = list(dict.fromkeys([*BASE_EXTRACTOR_SOURCES, *SOURCES,
        "scripts/run_finalist_pipeline_v2.py", "scripts/run_finalist_pipeline_v3.py", "finalist_v2.sh", "finalist_v3.sh"]))


def stage_command(run, name, attempt):
    return [sys.executable, str(Path(__file__).resolve()), "_stage", "--output-dir", str(run),
            "--stage", name, "--stage-dir", str(attempt)]


def prepare(args, qa, sample, plan):
    run = args.output_dir.resolve()
    if run.exists():
        raise FileExistsError("new partial acquisition output required; use --resume for its frozen plan")
    config = json.loads(args.config.read_text())
    if (config.get("kind") != KIND or config.get("schema_version") != 1
            or config["detector"] != {"device": "0", "probes": 8}):
        raise ValueError("partial acquisition requires its declared recipe and unchanged person crop settings")
    acquisition = read_acquisition(qa, args.scene_manifest)
    raw, modalities = collect_raw(args.raw_media_root.resolve(), config, plan)
    check_sensor_coverage(plan, modalities)
    profile = json.loads(args.profile_config.read_text())
    dependencies, model_sources, checkpoints = experts.checkpoint_inputs(profile, plan)
    dependencies += extraction_dependencies(config, plan)
    profile = {**profile, "checkpoints": {name: str(path) for name, path in checkpoints.items()},
               "features": {name: "extracted_during_this_run" for name in plan.features}}
    sources = [base.entry(path) for path in base.source_closure(
        [*base.EXTRACTOR_SOURCES, *[experts.ENTRYPOINTS[name] for name in plan.experts], *map(str, model_sources)])]
    run.mkdir(parents=True)
    for directory in ("inputs/qa", "source_snapshot", "stages", "guard", "read_audit"):
        (run / directory).mkdir(parents=True)
    qa.to_csv(run / "inputs/qa/test_qa.csv", index=False)
    sample.to_csv(run / "inputs/qa/sample_submission.csv", index=False)
    base.make_raw_view(run, raw, modalities)
    base.write(run / "raw_input_manifest.json", raw)
    (run / "guard/sitecustomize.py").write_text(base.GUARD_SOURCE)
    for record in sources:
        target = run / "source_snapshot" / Path(record["path"]).relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(record["path"], target)
    input_paths = [args.config, args.profile_config, args.qa_dir / "test_qa.csv", args.qa_dir / "sample_submission.csv",
                   run / "inputs/qa/test_qa.csv", run / "inputs/qa/sample_submission.csv",
                   run / "raw_view/sensors/manifest_nonvisual.csv", run / "guard/sitecustomize.py",
                   run / "raw_input_manifest.json"]
    acquisition_record = None
    if acquisition is not None:
        path = run / "inputs/acquisition.csv"
        acquisition.scenes.to_csv(path, index=False)
        acquisition_record = base.entry(path)
        input_paths += [args.scene_manifest, path]
    generic = {}
    if "emotion_visual" in plan.features:
        generic = json.loads(base.resolve(config["generic_config"]).read_text())
        for key in ("train_csv", "test_csv", "expected_training_units", "expected_training_qa_units"):
            generic.pop(key, None)
        generic.update(media_root=str(run / "raw_view/visual"), weights_path=str(base.resolve(config["generic_weights"])),
                       person_window_caches=[], scope="Current Testing HAU only")
    contract = {"kind": KIND, "created_utc": base.now(), "config": config, "profile": profile,
        "generic_config": generic, "batch_plan": plan.to_dict(), "units": list(plan.units),
        "hau_units": list(plan.hau_units), "harn_units": list(plan.harn_units),
        "qa_dir": str(run / "inputs/qa"), "raw_media_root": str(args.raw_media_root.resolve()),
        "requested_qa_dir": str(args.qa_dir.resolve()), "profile_config": str(args.profile_config.resolve()),
        "recipe_config": str(args.config.resolve()), "workers": args.workers,
        "requested_scene_manifest": str(args.scene_manifest.resolve()) if acquisition is not None else None,
        "acquisition": acquisition_record, "python": sys.executable, "python_prefix": sys.prefix,
        "inputs": [base.entry(path) for path in input_paths], "dependencies": dependencies, "sources": sources,
        "raw_files": len(raw), "raw_bytes": sum(item["bytes"] for item in raw), "stages": list(plan.stages),
        "stage_commands": {name: stage_command(run, name, run / "stages" / name / "attempt_01") for name in plan.stages},
        "versions": {name: importlib.metadata.version(name) for name in
                     ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "torch", "torchvision", "ultralytics")},
        "selection": "Experts and extraction stages chosen solely from current unlabelled source/category metadata"}
    base.write(run / "contract.json", contract)
    (run / "contract.sha256").write_text(base.sha(run / "contract.json") + "\n")
    base.write(run / "state.json", {"status": "prepared", "created_utc": base.now(), "stages": {}})
    print(json.dumps({"status": "prepared", "experts": list(plan.experts), "raw_files": len(raw),
                      "stages_pending": list(plan.stages)}, indent=2), flush=True)
    return contract


def feature_index(run, contract, state):
    paths, records = {}, {}
    for name in contract["batch_plan"]["features"]:
        path = base.stage_output(run, state, name)
        archive = path
        if name == "harn_middle":
            descriptor = json.loads(path.read_text())
            archive = path.parent / descriptor["units"]["path"]
        with np.load(archive, allow_pickle=False) as values:
            units = sorted(set(values["units"].astype(str)))
        required = contract["batch_plan"]["feature_units"][name]
        if not units or not set(units) <= set(required):
            raise ValueError(f"{name} features have empty or unrequested unit coverage")
        if name in {"harn_middle", "logits8", "logits16"} and units != required:
            raise ValueError(f"{name} did not cover every required unit")
        paths[name] = str(path)
        records[name] = {**base.entry(path), "units": units, "missing_units": sorted(set(required) - set(units))}
    return {**contract["profile"], "features": paths}, {"kind": "newly_extracted_partial_testing_features", "features": records}


def sensor_stage(name, run, attempt, contract):
    """Reuse the frozen extraction loop, restricting its manifest to needed units."""
    original = importlib.import_module("reproduce_raw_test_features")
    required = set(contract["batch_plan"]["feature_units"][name])
    old_pd, old_root = original.pd, original.SENSOR_ROOT

    def read_manifest(*args, **kwargs):
        frame = old_pd.read_csv(*args, **kwargs)
        return frame.loc[frame.unit.isin(required)].copy()

    try:
        original.pd = SimpleNamespace(read_csv=read_manifest)
        original.SENSOR_ROOT = run / "raw_view/sensors"
        return original.sensors(name, attempt, {**contract, "units": sorted(required)})
    finally:
        original.pd, original.SENSOR_ROOT = old_pd, old_root


def middle_stage(run, attempt, contract, state):
    """Use the frozen video extractor with an exact HARn unit filter."""
    original = importlib.import_module("extract_video_midfeatures")
    discover = original.discover_videos
    allowed = set(contract["harn_units"])

    def discover_harn(*args, **kwargs):
        return [row for row in discover(*args, **kwargs) if row[0] in allowed]

    output = attempt / "middle/manifest.json"
    spec = contract["config"]["middle"]
    previous = sys.argv
    try:
        original.discover_videos = discover_harn
        sys.argv = [str(ROOT / "scripts/extract_video_midfeatures.py"), "--media-root", str(run / "raw_view/visual"),
            "--weights", str(base.resolve(contract["config"]["generic_weights"])), "--windows-json",
            str(base.stage_output(run, state, "roi")), "--output-dir", str(output.parent),
            "--prefixes", "Testing/", "--batch-size", str(spec["batch_size"]), "--workers", str(spec["loader_workers"])]
        original.main()
        command = list(sys.argv)
    finally:
        original.discover_videos, sys.argv = discover, previous
    return output, {"original_entrypoint_argv": command, "exact_HARn_unit_filter": sorted(allowed)}


def execute_stage(run, name, attempt):
    contract, state = base.read_contract(run), json.loads((run / "state.json").read_text())
    if name not in contract["stages"]:
        raise ValueError("stage is not required by this frozen QA batch")
    if name not in {*base.SENSORS, "harn_middle", "inference"}:
        return BASE_EXECUTE_STAGE(run, name, attempt)
    (attempt / "features").mkdir()
    if name in base.SENSORS:
        result = sensor_stage(name, run, attempt, contract)
        output = Path(result["output"])
    elif name == "harn_middle":
        output, result = middle_stage(run, attempt, contract, state)
    else:
        profile, index = feature_index(run, contract, state)
        config_path = attempt / "inference_config.json"
        base.write(config_path, profile)
        base.write(attempt / "feature_index.json", index)
        output = attempt / "submission.csv"
        command = [sys.executable, str(ROOT / "scripts/infer_acquisition_experts_v2.py"), contract["qa_dir"], str(output),
                   "--config", str(config_path), "--workers", str(contract["workers"])]
        if contract["acquisition"] is not None:
            command += ["--scene-manifest", contract["acquisition"]["path"]]
        subprocess.run(command, cwd=ROOT, check=True)
        result = {"command": command, "validation": validate_submission_files(
            Path(contract["qa_dir"]) / "test_qa.csv", Path(contract["qa_dir"]) / "sample_submission.csv", output).to_dict()}
    base.write(attempt / "result.json", {**result, "primary_output": str(output), "completed_utc": base.now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run"):
        command = commands.add_parser(name)
        for key in ("qa-dir", "raw-media-root", "output-dir"):
            command.add_argument(f"--{key}", type=Path, required=True)
        command.add_argument("--scene-manifest", type=Path)
        command.add_argument("--profile-config", type=Path,
                             default=ROOT / "configs/checkpoint_inference_emotion_v6_harn_seed_ensemble_v1.json")
        command.add_argument("--config", type=Path, default=ROOT / "configs/finalist_acquisition_v2.json")
        command.add_argument("--workers", type=int, choices=(1, 2), default=2)
        if name == "run":
            command.add_argument("--resume", action="store_true")
            command.add_argument("--stop-after", choices=STAGE_ORDER)
            command.add_argument("--submission-output", type=Path)
    stage = commands.add_parser("_stage")
    stage.add_argument("--output-dir", type=Path, required=True)
    stage.add_argument("--stage", choices=STAGE_ORDER, required=True)
    stage.add_argument("--stage-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.output_dir.resolve()
    if args.command == "_stage":
        contract = base.read_contract(run)
        _, _, plan = read_batch(Path(contract["qa_dir"]))
        if contract["kind"] != KIND or plan.to_dict() != contract["batch_plan"]:
            raise ValueError("partial stage scope differs from the frozen contract")
        configure(plan)
        execute_stage(run, args.stage, args.stage_dir.resolve())
        return 0
    qa, sample, plan = read_batch(args.qa_dir.resolve())
    args.scene_manifest = (args.scene_manifest or args.qa_dir / "acquisition.csv") if plan.hau_units else None
    if getattr(args, "stop_after", None) and args.stop_after not in plan.stages:
        raise ValueError("--stop-after names an inactive stage for this question batch")
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:args.workers])
    configure(plan)
    if getattr(args, "submission_output", None) and args.submission_output.exists():
        raise FileExistsError("submission output must be new")
    if getattr(args, "resume", False):
        contract = base.read_contract(run)
        requested = {"kind": KIND, "requested_qa_dir": str(args.qa_dir.resolve()),
            "raw_media_root": str(args.raw_media_root.resolve()), "profile_config": str(args.profile_config.resolve()),
            "recipe_config": str(args.config.resolve()), "workers": args.workers, "batch_plan": plan.to_dict(),
            "requested_scene_manifest": str(args.scene_manifest.resolve()) if plan.hau_units else None}
        if any(contract[key] != value for key, value in requested.items()):
            raise ValueError("resume arguments or selected experts differ from the frozen partial batch")
    else:
        prepare(args, qa, sample, plan)
    if args.command == "run":
        base.run_pipeline(run, stop_after=args.stop_after)
        if json.loads((run / "state.json").read_text())["status"] == "complete":
            path = run / "completion.json"
            completion = json.loads(path.read_text())
            completion.update(batch_plan=plan.to_dict(), scope="Current QA subset to complete, format-validated answers")
            completion["limitations"][0] = "HAU uses explicit acquisition metadata; HARn-only batches need no scene manifest."
            base.write(path, completion)
            if args.submission_output:
                args.submission_output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(run / "submission.csv", args.submission_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
