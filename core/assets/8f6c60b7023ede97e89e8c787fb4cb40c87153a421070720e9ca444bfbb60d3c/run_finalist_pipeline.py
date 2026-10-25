#!/usr/bin/env python3
"""Run the selected four experts from raw Testing media, with resumable stages.

The supplied profile declares checkpoints and feature names. Its previous feature
paths are never opened. Every feature is extracted here, and only this run's
expert outputs are used to build the submission. ``prepare`` hashes inputs and
records commands without claiming that extraction or inference has run.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "src")]

from cuhkx.inference import QUESTION_COLUMNS, question_metadata
from cuhkx.metadata import feature_unit
from cuhkx.submission import validate_submission_files

KIND = "cuhkx_raw_testing_equal_seed_finalist_v1"
SENSORS = ("imu", "skeleton", "radar", "windows16", "windows32", "windows48")
FEATURES = (*SENSORS, "logits8", "logits16", "emotion_visual", "harn_middle")
STAGES = (*SENSORS, "roi", "logits8", "logits16", "emotion_visual", "harn_middle", "inference")
EXTRACTOR_SOURCES = [
    "inference.sh", "scripts/run_finalist_pipeline.py", "scripts/reproduce_raw_test_features.py",
    "scripts/reproduce_raw_test_clean_cuda.py", "scripts/extract_imu_v2_features.py",
    "scripts/extract_skeleton_features.py", "scripts/extract_radar_features.py",
    "scripts/extract_skeleton_windows.py", "scripts/extract_kuno_temporal_logits.py",
    "scripts/extract_kuno_transfer_logits.py", "scripts/extract_generic_hau_windows.py",
    "scripts/pack_emotion_generic_visual.py", "scripts/extract_video_midfeatures.py",
    "scripts/extract_video_embeddings.py", "scripts/infer_seed_ensemble_experts.py",
    "scripts/infer_adapted_experts.py", "scripts/infer_experts.py",
]


def now():
    return datetime.now(UTC).isoformat()


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(path, value):
    def default(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=default) + "\n")
    temporary.replace(path)


def entry(path, expected=None):
    path = Path(path).resolve()
    actual = sha(path)
    if expected is not None and actual != expected:
        raise ValueError(f"declared dependency SHA256 differs: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": actual}


def raw_entry(path):
    path = Path(path).resolve()
    if any(part.lower() == "training" for part in path.parts):
        raise ValueError("a Testing media link resolves into Training")
    return entry(path)


def verify(entries):
    for item in entries:
        path = Path(item["path"])
        if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            raise ValueError(f"frozen input/output changed: {path}")


def resolve(path):
    return (ROOT / path).resolve()


def source_closure(names):
    """Record only the entrypoints and their project imports, not unrelated work."""
    pending, found = list(names), set()
    while pending:
        path = resolve(pending.pop())
        if path in found or not path.is_file():
            continue
        found.add(path)
        if path.suffix != ".py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module, *(f"{node.module}.{alias.name}" for alias in node.names)]
            for module in modules:
                relative = Path(*module.split(".")).with_suffix(".py")
                for base in (ROOT, ROOT / "src", ROOT / "scripts"):
                    candidate = base / relative
                    if candidate.is_file():
                        pending.append(str(candidate))
    return sorted(found)


def checkpoint_inputs(profile):
    """Follow model dependencies needed by the four frozen inference methods."""
    if (set(profile) != {"schema_version", "profile", "checkpoints", "features"}
            or profile["schema_version"] != 1 or profile.get("profile") != "independent_equal_seed_harn"
            or set(profile.get("checkpoints", {})) != {"action", "emotion", "sequence", "harn"}
            or set(profile.get("features", {})) != set(FEATURES)):
        raise ValueError("supply the complete emotion-v6 and equal-seed HARn profile")
    files, sources = {}, set()

    def add(path, expected=None):
        record = entry(path, expected)
        files[record["path"]] = record
        return Path(record["path"])

    def model_refs(descriptor, parent):
        refs = descriptor.get("models", {})
        for item in refs.values() if isinstance(refs, dict) else refs:
            path = add(parent / item["path"], item["sha256"])
            if path.suffix == ".json":
                nested = json.loads(path.read_text())
                collect_sources(nested)
                model_refs(nested, path.parent)

    def collect_sources(descriptor):
        for key in ("source_hashes", "recipe_source_hashes", "original_source_hashes"):
            for path, digest in descriptor.get(key, {}).items():
                sources.add(add(resolve(path), digest))
        for record in descriptor.get("sources", []):
            sources.add(add(record["path"], record["sha256"]))

    for name, value in profile["checkpoints"].items():
        path = add(resolve(value))
        descriptor = json.loads(path.read_text())
        collect_sources(descriptor)
        if name in ("action", "sequence"):
            sidecar = add(path.with_suffix(".json.sha256"))
            if sidecar.read_text().strip() != sha(path):
                raise ValueError("checkpoint descriptor sidecar differs")
        if name == "action":
            model_refs(descriptor, path.parent)
        elif name == "sequence":
            model = descriptor["model"]
            add(path.parent / model["path"], model["sha256"])
        elif name == "emotion":
            if descriptor.get("model_kind") != "cuhkx_emotion_recipe_nested_v6":
                raise ValueError("the raw finalist recipe expects emotion v6")
            sidecar = add(path.with_suffix(".sha256"))
            if sidecar.read_text().strip() != sha(path):
                raise ValueError("emotion inference policy sidecar differs")
            for model in [descriptor["source_manifest"], descriptor["source_model"]["checkpoint"],
                          descriptor["source_model"]["scope"]]:
                add(model["path"], model["sha256"])
        else:
            if (descriptor.get("kind") != "cuhkx_harn_equal_seed_ensemble_v1"
                    or descriptor.get("visual_weights") != [.5, .5]):
                raise ValueError("HARn must be the fixed equal two-seed checkpoint")
            for model in descriptor["classifiers"].values():
                add(model["path"], model["sha256"])
    return list(files.values()), sources


def read_questions(qa_dir, expected):
    qa_path, sample_path = qa_dir / "test_qa.csv", qa_dir / "sample_submission.csv"
    qa = question_metadata(pd.read_csv(qa_path, usecols=QUESTION_COLUMNS,
                                       dtype=str, keep_default_na=False))
    sample = pd.read_csv(sample_path, dtype=str, keep_default_na=False)
    if sample.columns.tolist() != ["qa_id", "prediction"] or sample.qa_id.tolist() != qa.qa_id.tolist():
        raise ValueError("sample schema, IDs or order differ from Testing questions")
    units = sorted({feature_unit(path, training=False) for path in qa.path})
    hau = sorted({feature_unit(path, training=False) for path in qa.loc[qa.source.eq("HAU"), "path"]})
    for unit in units:
        if (Path(unit).is_absolute() or ".." in Path(unit).parts
                or any(part.lower() == "training" for part in Path(unit).parts)):
            raise ValueError("QA media paths must remain within the Testing split")
    if {"questions": len(qa), "units": len(units), "hau_units": len(hau)} != expected:
        raise ValueError("QA scope differs from the declared finalist acquisition layout")
    return qa, sample, units, hau


def collect_raw(raw_root, config, units):
    """Inspect only the declared Testing units, never the sibling Training tree."""
    roots = {key: (raw_root / config["layout"][key]).resolve() for key in ("visual", "sensors")}
    records, modalities = [], []
    for unit in units:
        visual = roots["visual"] / unit
        for modality in ("Depth", "IR", "Depth_Color"):
            path = visual / modality / f"{modality}.mp4"
            if not path.is_file():
                if modality != "Depth_Color":
                    raise FileNotFoundError(f"the current frozen recipe requires paired Depth/IR: {path}")
                continue
            records.append({"kind": "visual", "relative": str(path.relative_to(roots["visual"])),
                            **raw_entry(path)})
        sensor = roots["sensors"] / unit
        present = []
        for modality in ("IMU", "Radar", "Skeleton"):
            files = sorted(path for path in (sensor / modality).rglob("*") if path.is_file())
            if files:
                present.append(modality)
            for path in files:
                records.append({"kind": "sensors", "relative": str(path.relative_to(roots["sensors"])),
                                **raw_entry(path)})
        modalities.append({"unit": unit, "modalities": ";".join(present)})
    return records, modalities


def make_raw_view(run, records, modalities):
    for record in records:
        target = run / "raw_view" / record["kind"] / record["relative"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(record["path"])
    sensor_root = run / "raw_view/sensors"
    sensor_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(modalities).to_csv(sensor_root / "manifest_nonvisual.csv", index=False)


def verify_raw_view(run, records):
    for record in records:
        link = run / "raw_view" / record["kind"] / record["relative"]
        if not link.is_symlink() or str(link.resolve()) != record["path"]:
            raise ValueError(f"Testing media view no longer targets the frozen raw input: {link}")
    verify(records)


def stage_command(run, name, attempt):
    return [sys.executable, str(Path(__file__).resolve()), "_stage", "--output-dir", str(run),
            "--stage", name, "--stage-dir", str(attempt)]


def stage_output(run, state, name):
    item = state["stages"].get(name)
    if not item or item["status"] != "complete":
        raise ValueError(f"required stage has not completed: {name}")
    return Path(item["primary_output"])


def feature_index(run, contract, state):
    paths = {name: str(stage_output(run, state, name)) for name in FEATURES}
    records = {}
    for name, path in paths.items():
        if name == "harn_middle":
            descriptor = json.loads(Path(path).read_text())
            archive = Path(path).parent / descriptor["units"]["path"]
        else:
            archive = Path(path)
        with np.load(archive, allow_pickle=False) as data:
            units = sorted(set(data["units"].astype(str)))
        if not units or not set(units) <= set(contract["units"]):
            raise ValueError(f"feature contains missing/extra/Training unit keys: {name}")
        required = contract["hau_units"] if name in ("logits8", "logits16") else None
        if name == "harn_middle":
            required = contract["units"]
        if required is not None and units != required:
            raise ValueError(f"HAU visual logits do not cover every required unit: {name}")
        records[name] = {**entry(path), "units": units,
                         "missing_units": sorted(set(contract["units"]) - set(units))}
    profile = {**contract["profile"], "features": paths}
    index = {"kind": "newly_extracted_testing_features", "features": records}
    return profile, index


GUARD_SOURCE = '''"""Propagated Testing-only read audit; generated by the finalist runner."""
import atexit
import json
import os
import sys
from pathlib import Path

root = Path(os.environ["CUHKX_FINALIST_RUN"]).resolve()
workspace = Path(os.environ["CUHKX_FINALIST_ROOT"]).resolve()
blocked = []

def audit(event, args):
    if event == "socket.connect":
        blocked.append("network connection")
        raise PermissionError("finalist inference uses local weights and does not access the network")
    if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    mode = args[1] if len(args) > 1 else None
    if isinstance(mode, str) and any(char in mode for char in "wax+"):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    training = path.name.lower() == "training_qa.csv" or "training" in [p.lower() for p in path.parts]
    old_csv = path.suffix.lower() == ".csv" and path.is_relative_to(workspace) and not path.is_relative_to(root)
    sensor = "Testing" in path.parts and ("IMU" in path.parts or "Radar" in path.parts)
    if training or (old_csv and not sensor):
        blocked.append(str(path))
        raise PermissionError("finalist inference prohibits Training and old prediction CSV inputs: " + str(path))

sys.addaudithook(audit)

def finish():
    target = root / "read_audit" / (str(os.getpid()) + ".json")
    target.write_text(json.dumps({"pid": os.getpid(), "blocked": blocked,
        "python": sys.executable, "python_prefix": sys.prefix, "argv": sys.argv,
        "cpu_affinity": sorted(os.sched_getaffinity(0))}) + "\\n")

atexit.register(finish)
'''


def prepare(args):
    run = args.output_dir.resolve()
    if run.exists():
        raise FileExistsError("new finalist output directory required; use run --resume for an existing plan")
    config = json.loads(args.config.read_text())
    if config.get("kind") != KIND or config.get("schema_version") != 1:
        raise ValueError("unsupported finalist raw extraction recipe")
    if config["detector"] != {"device": "0", "probes": 8}:
        raise ValueError("the verified model requires the original CUDA person crop recipe")
    profile = json.loads(args.profile_config.read_text())
    dependencies, model_sources = checkpoint_inputs(profile)
    qa, sample, units, hau = read_questions(args.qa_dir.resolve(), config["expected"])
    raw, modalities = collect_raw(args.raw_media_root.resolve(), config, units)
    for path, digest in config["kuno_files"].items():
        dependencies.append(entry(resolve(config["kuno_model_dir"]) / path, digest))
    dependencies.append(entry(resolve(config["generic_weights"]), config["generic_weights_sha256"]))
    dependencies += [entry(resolve(config[key])) for key in ("generic_config", "dependencies_lock")]
    sources = [entry(path) for path in source_closure([*EXTRACTOR_SOURCES, *map(str, model_sources)])]
    sources += [entry(ROOT / "finalist.sh")]
    run.mkdir(parents=True)
    for directory in ("inputs/qa", "source_snapshot", "stages", "guard", "read_audit"):
        (run / directory).mkdir(parents=True)
    qa.to_csv(run / "inputs/qa/test_qa.csv", index=False, lineterminator="\n")
    sample.to_csv(run / "inputs/qa/sample_submission.csv", index=False, lineterminator="\n")
    make_raw_view(run, raw, modalities)
    write(run / "raw_input_manifest.json", raw)
    (run / "guard/sitecustomize.py").write_text(GUARD_SOURCE)
    for record in sources:
        path = Path(record["path"])
        target = run / "source_snapshot" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    inputs = [entry(path) for path in [args.config, args.profile_config,
              args.qa_dir / "test_qa.csv", args.qa_dir / "sample_submission.csv",
              run / "inputs/qa/test_qa.csv", run / "inputs/qa/sample_submission.csv",
              run / "raw_view/sensors/manifest_nonvisual.csv", run / "guard/sitecustomize.py",
              run / "raw_input_manifest.json"]]
    generic = json.loads(resolve(config["generic_config"]).read_text())
    for key in ("train_csv", "test_csv", "expected_training_units", "expected_training_qa_units"):
        generic.pop(key, None)
    generic.update(media_root=str(run / "raw_view/visual"), weights_path=str(resolve(config["generic_weights"])),
                   person_window_caches=[], scope="Testing HAU only")
    contract = {"kind": KIND, "created_utc": now(), "config": config, "profile": profile,
                "generic_config": generic, "units": units, "hau_units": hau,
                "qa_dir": str(run / "inputs/qa"), "raw_media_root": str(args.raw_media_root.resolve()),
                "requested_qa_dir": str(args.qa_dir.resolve()), "profile_config": str(args.profile_config.resolve()),
                "recipe_config": str(args.config.resolve()), "workers": args.workers,
                "python": sys.executable, "python_prefix": sys.prefix, "inputs": inputs,
                "dependencies": dependencies, "sources": sources,
                "raw_files": len(raw), "raw_bytes": sum(item["bytes"] for item in raw),
                "stages": list(STAGES), "stage_commands": {name: stage_command(
                    run, name, run / "stages" / name / "attempt_01") for name in STAGES},
                "versions": {name: importlib.metadata.version(name) for name in
                    ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "torch", "torchvision", "ultralytics")}}
    write(run / "contract.json", contract)
    (run / "contract.sha256").write_text(sha(run / "contract.json") + "\n")
    write(run / "state.json", {"status": "prepared", "created_utc": now(), "stages": {}})
    print(json.dumps({"status": "prepared", "output_dir": str(run), "raw_files": len(raw),
                      "raw_bytes": contract["raw_bytes"], "stages_pending": len(STAGES)}), flush=True)
    return contract


def read_contract(run):
    path = run / "contract.json"
    if sha(path) != (run / "contract.sha256").read_text().strip():
        raise ValueError("finalist contract changed")
    return json.loads(path.read_text())


def generic_visual(attempt, run, contract, crops):
    from extract_generic_hau_windows import archive_arrays, embeddings, setup_runtime
    from reproduce_raw_test_clean_cuda import complete_arrays

    from cuhkx.generic_video_windows import (
        sample_preprocessed,
        temporal_alignment,
        video_metadata,
        window_timing,
    )
    config = {**contract["generic_config"], "person_window_caches": [str(crops)],
              "output_npz": str(attempt / "features/generic_windows.npz")}
    windows = json.loads(crops.read_text())["windows"]
    records, alignment = [], {}
    for unit in contract["hau_units"]:
        metadata = {}
        for modality in config["modalities"]:
            path = run / "raw_view/visual" / unit / modality / f"{modality}.mp4"
            meta = video_metadata(path) if path.is_file() else None
            metadata[modality] = meta
            records.append({"unit": unit, "modality": modality, "available": meta is not None,
                            "path": str(path) if meta is not None else None, "roi": windows[unit],
                            "sha256": sha(path) if meta is not None else None,
                            "metadata": meta, "timing": window_timing(meta, config) if meta is not None else None})
        alignment[unit] = temporal_alignment(metadata["Depth"], metadata["IR"], config)
    inputs = {"units": contract["hau_units"], "records": records, "alignment": alignment,
              "qa_required_units": {"test": contract["hau_units"]}}
    extraction = attempt / "generic_extraction"
    extraction.mkdir()
    write(extraction / "frozen_config.json", {"config": config, "created_utc": now(),
          "source_hashes": {str(Path(item["path"]).relative_to(ROOT)): item["sha256"] for item in contract["sources"]}})
    write(extraction / "input_manifest.json", inputs)
    arrays = complete_arrays(archive_arrays(inputs, config), config)
    model = setup_runtime(config)
    for index, record in enumerate(records):
        if record["available"]:
            tensor = sample_preprocessed(record["path"], record["roi"], config)
            arrays["X"][index * 8:(index + 1) * 8] = embeddings(model, tensor, config)
        if (index + 1) % 20 == 0 or index + 1 == len(records):
            print(f"generic_raw_videos={index + 1}/{len(records)}", flush=True)
    np.savez_compressed(config["output_npz"], **arrays)
    write(extraction / "report.json", {"status": "Testing_generic_windows_extracted",
          "sha256": sha(config["output_npz"]), "output": config["output_npz"]})
    output = attempt / "features/emotion_visual.npz"
    sys.argv = [str(ROOT / "scripts/pack_emotion_generic_visual.py"), "--windows", config["output_npz"],
                "--extraction-run", str(extraction), "--output", str(output),
                "--run-dir", str(attempt / "generic_aggregation")]
    importlib.import_module("pack_emotion_generic_visual").main()
    return output


def execute_stage(run, name, attempt):
    contract = read_contract(run)
    state = json.loads((run / "state.json").read_text())
    (attempt / "features").mkdir()
    if name in SENSORS:
        import reproduce_raw_test_features as original
        original.SENSOR_ROOT = run / "raw_view/sensors"
        result = original.sensors(name, attempt, contract)
        output = Path(result["output"])
    elif name == "roi":
        import cv2
        import torch
        from extract_kuno_transfer_logits import detect_windows, discover_units
        cv2.setNumThreads(1)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        records = discover_units(run / "raw_view/visual", ("Testing/",))
        if [record["unit"] for record in records] != contract["units"]:
            raise ValueError("raw IR coverage differs from requested Testing units")
        detector = contract["config"]["detector"]
        windows, stats = detect_windows(records, resolve(contract["config"]["kuno_model_dir"]) / "yolo11n.pt",
                                        detector["probes"], detector["device"])
        output = attempt / "person_windows.json"
        write(output, {"units": contract["units"], "windows": windows, "stats": stats})
        result = {"provenance": "actual YOLO inference on raw Testing IR", "stats": stats}
    elif name in ("logits8", "logits16"):
        import cv2
        import torch
        cv2.setNumThreads(1)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        spec = contract["config"]["logits"][name]
        output = attempt / "features" / f"{name}.npz"
        arguments = ["--media-root", str(run / "raw_view/visual"), "--model-dir",
                     str(resolve(contract["config"]["kuno_model_dir"])), "--output", str(output),
                     "--prefixes", "Testing/", "--unit-regex", "^(?:" + "|".join(map(re.escape, contract["hau_units"])) + ")$",
                     "--frames", str(spec["frames"]), "--segments", str(spec["segments"]),
                     "--window-seconds", str(spec["window_seconds"]), "--batch-size", str(spec["batch_size"]),
                     "--windows-json", str(stage_output(run, state, "roi")), "--device", "cuda"]
        sys.argv = [str(ROOT / "scripts/extract_kuno_temporal_logits.py"), *arguments]
        code = importlib.import_module("extract_kuno_temporal_logits").main()
        if code:
            raise RuntimeError(f"frozen temporal extractor exited {code}")
        result = {"original_entrypoint_argv": sys.argv}
    elif name == "emotion_visual":
        output = generic_visual(attempt, run, contract, stage_output(run, state, "roi"))
        result = {"aggregation": "verified generic eight-window mean/std recipe"}
    elif name == "harn_middle":
        output = attempt / "middle/manifest.json"
        spec = contract["config"]["middle"]
        sys.argv = [str(ROOT / "scripts/extract_video_midfeatures.py"), "--media-root", str(run / "raw_view/visual"),
                    "--weights", str(resolve(contract["config"]["generic_weights"])), "--windows-json",
                    str(stage_output(run, state, "roi")), "--output-dir", str(output.parent),
                    "--prefixes", "Testing/", "--batch-size", str(spec["batch_size"]),
                    "--workers", str(spec["loader_workers"])]
        importlib.import_module("extract_video_midfeatures").main()
        result = {"original_entrypoint_argv": sys.argv}
    elif name == "inference":
        profile, index = feature_index(run, contract, state)
        config_path = attempt / "inference_config.json"
        write(config_path, profile)
        write(attempt / "feature_index.json", index)
        output = attempt / "submission.csv"
        command = [sys.executable, str(ROOT / "scripts/infer_seed_ensemble_experts.py"), contract["qa_dir"],
                   str(output), "--config", str(config_path), "--workers", str(contract["workers"])]
        subprocess.run(command, cwd=ROOT, check=True)
        result = {"validation": validate_submission_files(Path(contract["qa_dir"]) / "test_qa.csv",
                    Path(contract["qa_dir"]) / "sample_submission.csv", output).to_dict(), "command": command}
    else:
        raise ValueError(f"unknown stage: {name}")
    write(attempt / "result.json", {**result, "primary_output": str(output), "completed_utc": now()})


def run_pipeline(run, stop_after=None):
    contract = read_contract(run)
    if sys.executable != contract["python"] or sys.prefix != contract["python_prefix"]:
        raise ValueError("resume with the same Python environment used to prepare the plan")
    verify([*contract["inputs"], *contract["dependencies"], *contract["sources"]])
    verify_raw_view(run, json.loads((run / "raw_input_manifest.json").read_text()))
    cpus = sorted(os.sched_getaffinity(0))[:contract["workers"]]
    os.sched_setaffinity(0, cpus)
    environment = dict(os.environ, PYTHONPATH=f"{run / 'guard'}:{ROOT}:{ROOT / 'src'}",
        CUHKX_FINALIST_RUN=str(run), CUHKX_FINALIST_ROOT=str(ROOT), PYTHONNOUSERSITE="1",
        OMP_NUM_THREADS=str(contract["workers"]), OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS=str(contract["workers"]), OMP_THREAD_LIMIT=str(contract["workers"]),
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", YOLO_OFFLINE="1")
    state_path = run / "state.json"
    state = json.loads(state_path.read_text())

    def pause_after(name):
        if name != stop_after or name == STAGES[-1]:
            return False
        state.update(status="paused", paused_after=name, active_stage=None)
        write(state_path, state)
        print(f"status=paused after={name}; resume explicitly to run remaining stages", flush=True)
        return True

    for name in STAGES:
        previous = state["stages"].get(name)
        if previous and previous["status"] == "complete":
            verify(previous["outputs"])
            print(f"stage={name} reused_after_sha256_check", flush=True)
            if pause_after(name):
                return
            continue
        number = 1 if previous is None else previous["attempt"] + 1
        attempt = run / "stages" / name / f"attempt_{number:02d}"
        attempt.mkdir(parents=True)
        command = stage_command(run, name, attempt)
        receipt = {"status": "running", "attempt": number, "directory": str(attempt),
                   "started_utc": now(), "command": command, "cpu_affinity": cpus}
        state.update(status="running", active_stage=name)
        state["stages"][name] = receipt
        write(state_path, state)
        print(f"stage={name} starting attempt={number}", flush=True)
        began = time.monotonic()
        with (attempt / "stage.log").open("x") as log:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                                    stderr=subprocess.STDOUT, check=False)
        receipt.update(completed_utc=now(), seconds=time.monotonic() - began, exit_code=result.returncode)
        if result.returncode:
            receipt["status"] = state["status"] = "failed"
            write(state_path, state)
            raise RuntimeError(f"stage {name} failed; see {attempt / 'stage.log'}")
        receipt.update(status="complete", primary_output=json.loads((attempt / "result.json").read_text())["primary_output"],
                       outputs=[entry(path) for path in sorted(attempt.rglob("*")) if path.is_file() and path.suffix != ".log"])
        write(state_path, state)
        print(f"stage={name} complete seconds={receipt['seconds']:.1f}", flush=True)
        if pause_after(name):
            return
    verify([*contract["inputs"], *contract["dependencies"], *contract["sources"]])
    verify_raw_view(run, json.loads((run / "raw_input_manifest.json").read_text()))
    audits = [json.loads(path.read_text()) for path in (run / "read_audit").glob("*.json")]
    if not audits or any(record["blocked"] for record in audits):
        raise ValueError("a pipeline process attempted a prohibited Training/old CSV read or network connection")
    actual = stage_output(run, state, "inference")
    destination = run / "submission.csv"
    if destination.exists():
        if sha(destination) != sha(actual):
            raise ValueError("completed submission changed")
    else:
        shutil.copyfile(actual, destination)
    validation = validate_submission_files(Path(contract["qa_dir"]) / "test_qa.csv",
                  Path(contract["qa_dir"]) / "sample_submission.csv", destination).to_dict()
    write(run / "feature_index.json", json.loads((actual.parent / "feature_index.json").read_text()))
    completion = {"status": "raw_testing_to_complete_submission", "completed_utc": now(),
        "contract_sha256": sha(run / "contract.json"), "submission": entry(destination),
        "format_validation": validation, "feature_index_sha256": sha(run / "feature_index.json"),
        "completed_stages": len(STAGES), "raw_files_rehashed": contract["raw_files"],
        "raw_bytes_rehashed": contract["raw_bytes"], "audited_processes": len(audits),
        "blocked_reads": 0, "training_or_downloads_or_uploads": False,
        "limitations": ["Uses the original official acquisition layout and supplied trained model descriptors.",
                        "This run verifies inference reproduction, not model accuracy or full training reproduction."]}
    write(run / "completion.json", completion)
    state.update(status="complete", completed_utc=now(), active_stage=None)
    write(state_path, state)
    print(json.dumps(completion, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run"):
        command = sub.add_parser(name)
        command.add_argument("--qa-dir", type=Path, required=True)
        command.add_argument("--raw-media-root", type=Path, required=True,
                             help="contains visual/Testing and nonvisual/LMT_(IMU,Radar,Skeleton)/Testing")
        command.add_argument("--profile-config", type=Path, required=True)
        command.add_argument("--config", type=Path, default=ROOT / "configs/finalist_pipeline_v1.json")
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--workers", type=int, choices=(1, 2), default=2)
        if name == "run":
            command.add_argument("--resume", action="store_true")
            command.add_argument("--stop-after", choices=STAGES,
                                 help="pause after a fully recorded stage; no background work remains")
    stage = sub.add_parser("_stage", help=argparse.SUPPRESS)
    stage.add_argument("--output-dir", type=Path, required=True)
    stage.add_argument("--stage", choices=STAGES, required=True)
    stage.add_argument("--stage-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.output_dir.resolve()
    if args.command == "_stage":
        execute_stage(run, args.stage, args.stage_dir.resolve())
        return 0
    if getattr(args, "resume", False):
        contract = read_contract(run)
        requested = {"requested_qa_dir": str(args.qa_dir.resolve()), "raw_media_root": str(args.raw_media_root.resolve()),
                     "profile_config": str(args.profile_config.resolve()), "recipe_config": str(args.config.resolve()),
                     "workers": args.workers}
        if any(contract[key] != value for key, value in requested.items()):
            raise ValueError("resume arguments differ from the frozen plan")
    else:
        prepare(args)
    if args.command == "run":
        run_pipeline(run, stop_after=args.stop_after)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
