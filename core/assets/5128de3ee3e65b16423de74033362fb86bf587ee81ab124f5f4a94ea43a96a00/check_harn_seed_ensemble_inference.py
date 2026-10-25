#!/usr/bin/env python3
"""Replay both actual adapted models with Testing-only inputs and poisoned targets."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from cuhkx.harn_nested import sha256_file, write_json
from cuhkx.harn_seed_ensemble import NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "test-csv", "skeleton", "visual", "reference-predictions", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    middle = json.loads(args.visual.read_text())
    array_paths = [args.skeleton.resolve(), (args.visual.parent / middle["features"]["path"]).resolve(),
                   (args.visual.parent / middle["units"]["path"]).resolve()]
    unit_counts = {}
    for path in (array_paths[0], array_paths[2]):
        with np.load(path, allow_pickle=False) as archive:
            units = archive["units"].astype(str)
        if not len(units) or not np.char.startswith(units, "Testing/").all():
            raise ValueError("guard inputs must contain only Testing feature units")
        unit_counts[str(path)] = len(units)
    checkpoint = json.loads(args.checkpoint.read_text())
    model_paths = [Path(checkpoint["classifiers"][name]["path"]).resolve() for name in NAMES]
    inputs = {str(path.resolve()): sha256_file(path) for path in
              [args.checkpoint, args.test_csv, args.skeleton, args.visual, args.reference_predictions,
               Path(__file__), *array_paths, *model_paths]}
    write_json(out / "frozen_inputs.json", {"created_utc": datetime.now(UTC).isoformat(), "inputs": inputs})
    qa = pd.read_csv(args.test_csv, dtype=str, keep_default_na=False)
    qa.to_csv(out / "ordinary_qa.csv", index=False)
    poisoned = qa.copy()
    poisoned["qa_id"] = [f"equal_seed_fresh_qa_{index:04d}" for index in range(len(qa))]
    restore = dict(zip(poisoned.qa_id, qa.qa_id, strict=True))
    for name in ("answer", "prediction", "truth_word", "subject", "label", "action_class"):
        poisoned[name] = "POISON_NEVER_CONSUMED"
    poisoned.to_csv(out / "poisoned_qa.csv", index=False)
    wrapper = out / "guarded_infer.py"
    wrapper.write_text('''import json,sys
from pathlib import Path
import pandas as pd,numpy as np,joblib
root=Path(sys.argv[1]);sys.path[:0]=[str(root),str(root/'src')]
from scripts import run_harn_seed_ensemble as cli
job=json.loads(Path(sys.argv[2]).read_text())
reads={'csv':[],'arrays':[],'models':[]}
old_csv=pd.read_csv;old_np=np.load;old_model=joblib.load
def gate(kind,path,allowed):
    path=str(Path(path).resolve());reads[kind].append(path)
    if path not in allowed:raise RuntimeError('Unexpected '+kind+' input '+path)
def csv(path,*args,**kwargs):
    gate('csv',path,[job['qa']]);return old_csv(path,*args,**kwargs)
def arrays(path,*args,**kwargs):
    gate('arrays',path,job['arrays']);return old_np(path,*args,**kwargs)
def models(path,*args,**kwargs):
    gate('models',path,job['models']);return old_model(path,*args,**kwargs)
pd.read_csv=csv;np.load=arrays;joblib.load=models
sys.argv=job['arguments']
try:cli.main()
finally:Path(job['reads']).write_text(json.dumps(reads,indent=2)+'\\n')
''')
    environment = os.environ.copy()
    environment.update(PYTHONPATH=f"{ROOT}:{ROOT / 'src'}", OMP_NUM_THREADS="3", OPENBLAS_NUM_THREADS="3", MKL_NUM_THREADS="3")
    execution, reads = [], {}
    for name in ("ordinary", "poisoned"):
        csv = out / f"{name}_qa.csv"
        arguments = [str(ROOT / "scripts/run_harn_seed_ensemble.py"), "infer", "--checkpoint", str(args.checkpoint.resolve()),
                     "--qa-csv", str(csv), "--skeleton", str(args.skeleton.resolve()), "--visual", str(args.visual.resolve()),
                     "--output-dir", str(out / name)]
        job = {"qa": str(csv), "arrays": [str(path) for path in array_paths], "models": [str(path) for path in model_paths],
               "arguments": arguments, "reads": str(out / f"{name}_reads.json")}
        job_path = out / f"{name}_job.json"
        write_json(job_path, job)
        with (out / f"{name}.log").open("w") as log:
            process = subprocess.run([sys.executable, str(wrapper), str(ROOT), str(job_path)], cwd=ROOT,
                                     env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
        execution.append({"mode": name, "exit_code": process.returncode, "job_sha256": sha256_file(job_path)})
        write_json(out / "execution_status.json", execution)
        if process.returncode:
            raise RuntimeError(f"actual two-adapted-model {name} inference failed")
        reads[name] = json.loads((out / f"{name}_reads.json").read_text())
        if (reads[name]["csv"] != [str(csv)] or reads[name]["models"] != [str(path) for path in model_paths]
                or set(reads[name]["arrays"]) != {str(path) for path in array_paths}):
            raise ValueError("actual inference readers did not use exactly the declared two models/Testing archives")
    expected = pd.read_csv(args.reference_predictions, keep_default_na=False)
    for name in ("ordinary", "poisoned"):
        actual = pd.read_csv(out / name / "test_predictions.csv", keep_default_na=False)
        if name == "poisoned":
            actual.qa_id = actual.qa_id.map(restore)
            if actual.qa_id.isna().any():
                raise ValueError("unknown fresh QA ID in inference output")
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=False, rtol=0, atol=1e-6)
    for path, digest in inputs.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError("guard input changed during actual inference")
    shutil.copyfile(Path(__file__), out / "source_snapshot.py")
    write_json(out / "report.json", {"status": "actual_two_adapted_models_Testing_only_new_ID_poison_replay_passed",
                                    "qa_metadata_rows": len(qa), "predicted_rows": len(expected),
                                    "training_feature_units": 0, "testing_unit_counts": unit_counts,
                                    "actual_reads": reads, "execution": execution, "new_neural_fits": 0,
                                    "completed_utc": datetime.now(UTC).isoformat(), "float_tolerance": 1e-6})
    print("Both adapted models and poisoned metadata replayed", len(expected), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
