# Reproduction

## Version

The public implementation comes from the submitted verification package. `inference.sh` selects **56239668** by default. Pass `56258762` as the third argument for the other selected version. Later experimental candidates and the unsubmitted compatibility adapter are not selected by this entry.

The public export retains executed-source snapshots without requiring Git-history archiving in raw visual preparation, sensor preparation, and source-window crop detection. It also removes a research-time write restriction tied to the original private workspace path. These changes permit execution from a GitHub ZIP, a Docker build without `.git`, and a caller-chosen output directory. Model tensors, preprocessing parameters, fusion settings, grouping, and answer decoding are unchanged. The bundled descriptors are updated where they list these files. A training helper, its test fixture and test configuration omitted from the inference-only export were also restored.

## Input contract

`test_qa.csv` contains `qa_id,source,path,category,question,A,B,C,D`. Each row asks one question about the referenced recording. The output format follows the question category: single-choice questions use one option letter, HAU `multi` questions use the selected letters in canonical order, and HAU `sequence` questions use a permutation of the four action letters. For example, `AC` selects options A and C, while `BDAC` orders B before D before A before C. A `combination` question selects one option that describes an action combination.

The input directory must contain the official visual and nonvisual Testing trees. `test_qa.csv` may be at the input root, under `Testing/`, or under `kaggle/`. A sibling `sample_submission.csv` is optional. If absent, the entry uses the QA order.

For HAU, the selected method uses related questions and three-recording acquisition groups. Published consecutive Test names provide that grouping. For names that carry the original `HAU/user.../...-1`, `...-2`, `...-3` convention, those suffixes provide the positions. Equivalent supplied metadata can be represented by a sibling file:

```csv
clip,group,position
LM_test_0193,acquisition_001,0
LM_test_0194,acquisition_001,1
LM_test_0195,acquisition_001,2
```

This is a schema example. Group entries must describe the actual collection. Anonymous recordings without this relationship require a different context policy; assigning arbitrary triplets does not reproduce the selected method. The original-Test score and reproduction result apply to the published collection.

## Native environment

Docker uses the pinned PyTorch image and packages in `environment/`. To construct a corresponding native environment with Python 3.11 or 3.12:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install --no-deps -r environment/requirements-docker.txt
python tools/download_models.py
bash inference.sh /absolute/path/to/official_test outputs/primary.csv
python tools/compare_predictions.py outputs/primary.csv
```

The runtime uses OpenCV's FFmpeg backend to decode MP4 files. On Linux, install `libglib2.0-0` and `libgl1` if your image does not provide them. The process expects a free CUDA device when it starts. It is a single-GPU pipeline; no tensor parallelism is used.

`CUHKX_PYTHON=/path/to/python bash inference.sh ...` selects a specific interpreter. The entry fixes the Python hash seed, BLAS thread counts, offline Hugging Face settings, and CUBLAS workspace convention. Video branches retain their original FP16/BF16/FP32 settings.

## Results and scope

| Validation | Profile | Complete raw-data run | Time | Original Test match |
|---|---|---|---:|---|
| RTX 3060 Laptop 6 GB, original offline Docker | Primary | Yes | 34.1 min | 682/682; exact CSV bytes |
| RTX 3060 Laptop 6 GB, original offline Docker | Secondary | Yes | 35.6 min | 682/682; exact CSV bytes |
| RTX 4090 24 GB, original native environment | Primary | Yes | 18.6 min | 682/682; exact CSV bytes |

Original measurements are in `reproduction/original_validation.json`. The 4090 secondary measurement in the private development archive reused freshly produced shared features, so it is not listed as a full-run benchmark here. Measured whole-device GPU memory on the original primary 3060 run peaked at 3729 MiB; sampled container RAM peaked at 8.27 GiB.

`reproduction/reference/final_submission_1.csv` is the primary reference; `_2.csv` is secondary. Inference does not read them. The comparison tool checks IDs, per-row answers, order, and exact bytes, and exits nonzero on a mismatch.

### Public release check — 25 October 2026

The public source was validated in a native Python 3.11 environment on the RTX 3060 Laptop:

| Check | Input and execution | Result |
|---|---|---|
| Primary | All original Test features regenerated from raw input; downstream stages resumed after packaging fixes | 682/682; exact CSV bytes |
| Secondary | Reused those fresh shared features; recomputed the secondary HARn and temporal branches and final decoding | 682/682; exact CSV bytes |
| Source-archive entry | Fresh `bash inference.sh` run without `.git`; 17 questions, four recordings, all seven task categories | 17/17 matching reference answers; uninterrupted entry run |
| CPU component tests | Partial supervision, held-out-person separation, explicit layout and model/fusion checks | 17 passed |

This release check did not rebuild the Docker image or measure a new full-run latency. The original Docker and native timings above retain their original scope. Environment versions and comparison results are in [public_release_validation.json](../reproduction/public_release_validation.json).

## Logs

The top-level work directory retains a log per stage. `baseline.log` covers raw feature production and the shared specialists; `source_presence.log` covers the external-source action branch; `selected_action.log` and `selected_sat.log` cover final answer composition. If a subprocess fails, inspect its stage log and the logs in `baseline/`. Start a new end-to-end run with a new output filename.

## Component checks

The included CPU tests exercise partial action supervision, held-out-person
separation, explicit acquisition layout, model replay and missing-view HARn
fusion. To run them from an environment with pytest:

```bash
cd core
PYTHONPATH=.:src python -m pytest -q tests
```
