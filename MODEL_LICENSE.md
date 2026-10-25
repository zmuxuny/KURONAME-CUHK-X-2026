# Model and training-record terms

The root Apache-2.0 license covers original KURONAME source code. **It does not relicense the model payloads or benchmark data.**

The public model release contains the frozen fitted states needed to reproduce submissions 56239668 and 56258762. It includes our learned heads and adapted parameters, public pretrained weights, and the descriptor metadata that binds them. It does not include raw challenge media, raw Charades videos, frame caches, sensor recordings, or QA tables.

| Files | Applicable terms |
|---|---|
| KURONAME fitted states in `artifacts/` and `core/assets/` | Released by Zeyang Cao and Yanqing Zhou for noncommercial research under CC BY-NC 4.0, to the extent of the authors' rights. Underlying dataset, initialization and third-party terms continue to apply. |
| Training-derived metadata in `training_configs/`, `training/` and fitted descriptors | CC BY-NC 4.0, with CUHK-X attribution and underlying data terms retained. |
| `ensemble_packed.pt` | Original K-KUNO competition/noncommercial research terms in [K_KUNO_LICENSES.md](core/delivery/licenses/K_KUNO_LICENSES.md). Credit phuongncn. |
| `yolo11n.pt` | Ultralytics AGPL-3.0; not Apache-2.0 or CC BY-NC. |
| `pytorch_model.bin` for VideoMAE | Original CC BY-NC 4.0 model card. |
| Generic R34 / R18 initialization weights | Original release terms and notices in `core/delivery/`; no additional rights to pretraining data are granted. |

Identical files can appear at both a readable pretrained path and a content-addressed asset path because the submitted descriptors use both layouts. Their license is the same in either location.

For the CC BY-NC 4.0 terms, see [the license](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en). Credit the authors, link the relevant source and license, and identify modifications. CUHK-X use remains subject to its [data license](core/delivery/licenses/CUHK_X_LICENSE_v2.txt). Commercial rights to CUHK-X materials are reserved by the owner. The research release does not grant commercial permission for the complete pipeline.
