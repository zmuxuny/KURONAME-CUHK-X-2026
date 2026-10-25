# Model and data provenance

The current 0.91228 profile computes every answer from saved models. No competitor
prediction table is an inference input. The asset inventory in `model_sources.json`
records the actual files and hashes used by the completed raw-data reproduction.
Model binaries and competition data are excluded from this Git repository.

| Asset | Use | Source evidence |
|---|---|---|
| K-KUNO E290 packed R(2+1)D-34 ensemble | Current HAU action and sequence visual features | Public release model card and license notices, copied in `licenses/`; its two training-subject fold memberships remain unverified |
| Torchvision R(2+1)D-18 Kinetics400 V1 | Current generic emotion features and the initialization of our two HARn adapted models | Torchvision source and weight identity; BSD-3-Clause package notice; its README separately discusses possible pretrained-weight dataset terms |
| IG65M→Kinetics400 clip32 R(2+1)D-34 | New generic-feature and subject-excluded fitting experiments | Author's 2019 release, full weight SHA, pinned architecture source and MIT notice; original VMZ Apache-2.0 notice/model-zoo lineage retained |
| Ultralytics YOLO11n | Raw-media person crop estimation | Release asset hash and publisher AGPL-3.0 notice |
| Official CUHK-X Large Model training and test data | Training labels and raw sensor/video inputs | Organizer CUHK-X License v2.0; competition/data terms and required attribution apply; data are not redistributed |
| Our fitted classifiers and decoding policies | Action/Skeleton calibration, emotion context, temporal decoding and HARn adaptation | Individual checkpoint descriptors, source hashes, training-scope records and nested audits under the ignored local `runs/` tree |

The notices describe their named components. A software repository's license is not
treated as proof of blanket rights over all datasets or all pretrained weights. The
K-KUNO card identifies a Small-Track-trained release for competition/noncommercial
research; it does not provide the upstream fold membership needed to certify our
historical HAU validation as subject-excluded. The new R34 branch begins from a
generic 2019 initialization and retains all CUHK fitting scopes separately.

Primary sources were retrieved on 2026-09-06 UTC and frozen in
`runs/20260906_finalist_notices_v1/source_receipts.json` and
`runs/20260906_r34_asset_acquisition_v1/completion.json`:

- [IG65M PyTorch release](https://github.com/moabitcoin/ig65m-pytorch/releases/tag/v1.0.0)
- [VMZ original model zoo](https://github.com/facebookresearch/VMZ/blob/35cdbd7a569fa0f0c4b139934cd871c4314d2119/c2/tutorials/model_zoo.md)
- [Torchvision 0.26 README](https://github.com/pytorch/vision/blob/v0.26.0/README.md)
- [Ultralytics license source](https://github.com/ultralytics/ultralytics/blob/v8.4.138/LICENSE)
- [Organizer competition and finalist requirements](https://openaiotlab.github.io/CUHK-X-Challenge/)
- [CUHK-X data license](https://openaiotlab.github.io/CUHK-X-Challenge/Rule/LICENSE.txt)

This inventory is a provenance record for the working package. It does not declare
the final checkpoint collection ready for unrestricted redistribution or replace the
organizer's eventual finalist checks.
