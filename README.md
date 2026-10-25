# KURONAME

**Task-Specific Evidence and Structured Decoding for Non-RGB Activity Question Answering**

CUHK-X Challenge 2026, Large Model Track · **Runner-up (亚军)** · **Most Popular Award**

Zeyang Cao · Yanqing Zhou  
Zhejiang University · The Chinese University of Hong Kong, Shenzhen  
Faculty advisor: Xiaoxue Ren, Zhejiang University — Best Advisor Award

[Technical report](report/KURONAME_Technical_Report.pdf) · [Method](docs/METHOD.md) · [Training](TRAINING.md) · [Reproduction](docs/REPRODUCTION.md) · [Model terms](MODEL_LICENSE.md)

KURONAME answers questions about actions, event order, execution style, and activity–object relations from depth, infrared, skeleton, IMU and radar observations. It uses specialized predictors to estimate the evidence each task needs, then maps that evidence to the supplied answer options. A Charades-adapted video encoder provides an additional action-presence signal; temporal models preserve window order; a media–text matching model estimates execution style. Structured decoders use the relations between questions and recordings.

This repository releases the code and fitted models of our two selected competition submissions. **The default is 56239668**, the primary system described in the report. The scores below are the original Kaggle Test scores; the awards above refer to the final competition outcome.

| Profile | Selected submission | Private accuracy | Public accuracy | Configuration |
|---|---|---:|---:|---|
| **Primary / default** | **56239668** | **0.90294** | **0.94444** | [Primary](configs/submission_56239668.json) |
| Secondary | 56258762 | 0.90000 | 0.94444 | [Secondary](configs/submission_56258762.json) |

## Installation

Use Linux x86_64, an NVIDIA Ampere-or-newer GPU, and an NVIDIA driver compatible with CUDA 12.8. Our original full runs used one RTX 3060 Laptop GPU with 6 GB VRAM and approximately 16 GB host RAM. Allow at least 25 GB of additional free space for weights, downloads and inference intermediates, plus the dataset and Docker image.

```bash
git clone https://github.com/zmuxuny/KURONAME-CUHK-X-2026.git
cd KURONAME-CUHK-X-2026
python3 tools/download_models.py
docker build -f environment/Dockerfile -t kuroname:public-v1 .
```

The download command retrieves the frozen runtime models from the public release. Read [MODEL_LICENSE.md](MODEL_LICENSE.md) for their individual terms. The Docker build installs pinned dependencies and includes the models. Neither the model download nor the build downloads the CUHK-X dataset.

For a native Python environment, see [docs/REPRODUCTION.md](docs/REPRODUCTION.md).

## Dataset

Obtain the official Large Model Track data through the [CUHK-X dataset access page](https://aiot-public-dataset-cuhk-x.cuhkaiot.com/) and the [challenge](https://openaiotlab.github.io/CUHK-X-Challenge/). This repository does not redistribute the recordings, sensor streams or QA tables.

Arrange the original Test data as follows. The directory named `visual` contains the published visual Testing tree; the nonvisual tree retains its original name.

```text
official_test/
├── test_qa.csv
├── sample_submission.csv                 # optional
├── visual/
│   └── Testing/LM_test_XXXX/...
└── nonvisual/
    └── LMT_(IMU,Radar,Skeleton)/
        └── Testing/LM_test_XXXX/...
```

The selected method uses questions from the same recording and the published **three-recording HAU acquisition groups**. For the original consecutive `LM_test_*` naming, the entry reconstructs those groups. Explicit equivalent metadata can be supplied in `acquisition.csv` with columns `clip,group,position`, where position is 0, 1 or 2. See [the input contract](docs/REPRODUCTION.md#input-contract) before using a different collection.

## Inference

From the repository root:

```bash
mkdir -p outputs
docker run --rm --gpus all --network none --shm-size 16g \
  -v /absolute/path/to/official_test:/input:ro \
  -v "$PWD/outputs":/output \
  kuroname:public-v1 bash inference.sh /input /output/primary.csv
```

To run the second selected submission, append `56258762` and use a fresh output filename:

```bash
docker run --rm --gpus all --network none --shm-size 16g \
  -v /absolute/path/to/official_test:/input:ro \
  -v "$PWD/outputs":/output \
  kuroname:public-v1 bash inference.sh /input /output/secondary.csv 56258762
```

Each run reads the supplied raw data, extracts its features, and writes `qa_id,prediction` in the input order. Logs and intermediates are retained in `<output_csv>.work/`. Use a fresh output name when starting a new run. No training, external API or internet connection is needed during inference.

Compare a completed original-Test run with the selected submission:

```bash
python3 tools/compare_predictions.py outputs/primary.csv
# Secondary:
python3 tools/compare_predictions.py outputs/secondary.csv --submission 56258762
```

The original submitted Docker package reproduced **682/682 predictions and exact CSV bytes** on the RTX 3060 Laptop: 34.1 minutes for the primary and 35.6 minutes for the secondary. A separate full **native** primary run on an RTX 4090 took 18.6 minutes. These are measurements from the submitted environment. [Reproduction records](reproduction/original_validation.json) distinguish the hardware and execution mode; rebuilding a Docker image is distinct from loading the original tested image.

## Repository layout

| Location | Contents |
|---|---|
| `inference.sh`, `src/` | Public input adapter and final composition |
| `core/scripts/`, `core/src/cuhkx/` | Feature extraction, training code, fitted estimator classes and decoding |
| `configs/` | The two selected profiles |
| `core/assets/`, `artifacts/`, `core/pretrained/` | Frozen descriptors and model locations; binary files are restored by the downloader |
| `training/`, `training_configs/` | Training recipes and archived fit records |
| `reproduction/` | Original reference predictions and measurements |
| `report/` | Public technical report and LaTeX source |

The source tree retains the module names required by serialized estimators. Descriptor paths such as `core/assets/<digest>/...` are the original runtime layout. Historical absolute paths inside descriptors are identifiers that the runtime binder resolves to bundled files; the original development workspace is not required.

## License and citation

Original KURONAME code is released under **Apache-2.0**. The report is **CC BY 4.0**. Third-party code, weights, and data retain their own terms; see [THIRD_PARTY.md](THIRD_PARTY.md) and [MODEL_LICENSE.md](MODEL_LICENSE.md). In particular, the full pipeline uses Ultralytics under AGPL-3.0 and research-only model/data resources; the repository's code license does not grant unrestricted commercial use of that combined runtime.

```bibtex
@techreport{cao2026kuroname,
  title  = {KURONAME: Task-Specific Evidence and Structured Decoding
            for Non-RGB Activity Question Answering},
  author = {Cao, Zeyang and Zhou, Yanqing},
  year   = {2026},
  institution = {Zhejiang University and
                 The Chinese University of Hong Kong, Shenzhen},
  url    = {https://github.com/zmuxuny/KURONAME-CUHK-X-2026}
}
```

Please also cite the CUHK-X dataset and the upstream methods used in your work. References and attribution are included in the technical report and `core/delivery/`.
