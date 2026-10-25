# Third-party sources

KURONAME's original implementation is Apache-2.0. This file identifies external components used by the selected solution. The licenses below apply to those components; they are not replaced by the repository license.

| Resource | Use | Source and terms |
|---|---|---|
| K-KUNO R(2+1)D-34 ensemble | Fixed HAU action and temporal evidence | [Model](https://www.kaggle.com/datasets/phuongncn/k-kuno-lb-0-711-r2plus1d-model), [notebook](https://www.kaggle.com/code/phuongncn/lb-0-711-yolo-person-crop-r2plus1d-100mb). Notebook code: Apache-2.0; checkpoint: competition / noncommercial research under the upstream notice. |
| IG65M PyTorch R34 | Architecture and IG65M-to-Kinetics initialization | [Implementation](https://github.com/moabitcoin/ig65m-pytorch), commit `fc749e2ee354c3e4ddbb144cf511bb868b008f61`, MIT. [Model release](https://github.com/moabitcoin/ig65m-pytorch/releases/tag/v1.0.0); original VMZ notice retained. |
| Torchvision R(2+1)D-18 | Generic video features and HARn heads | [Torchvision](https://github.com/pytorch/vision), BSD-3-Clause source. Pretraining-data terms are separate. |
| VideoMAE Base SSv2 | Generic video evidence | [Model](https://huggingface.co/MCG-NJU/videomae-base-finetuned-ssv2), revision `e3f23b7a9bcc6f6b0bbf44da097c0f94548c6ec5`, **CC BY-NC 4.0** model card. |
| Ultralytics YOLO11n | Person crop detection | [Source v8.4.138](https://github.com/ultralytics/ultralytics/tree/v8.4.138), [model assets](https://github.com/ultralytics/assets), **AGPL-3.0**. |
| Video Depth Anything Small | Synthetic depth for secondary source training | [Model](https://huggingface.co/depth-anything/Video-Depth-Anything-Small), revision `256875362cff76724b920335dfb4b29dd611f66e`; code `4f5ae23172ba60fd7bc11ef671cca678842c7072`; Apache-2.0 recorded at acquisition. Not an inference dependency. |
| Charades | External training supervision | [Official project](https://prior.allenai.org/projects/charades). Obtain videos under the original dataset terms. Source videos and generated depth are not distributed here. |
| CUHK-X | Challenge training and evaluation | [Dataset](https://aiot-public-dataset-cuhk-x.cuhkaiot.com/), [license](core/delivery/licenses/CUHK_X_LICENSE_v2.txt). Noncommercial research; raw data are not redistributable. |

Copies of the available upstream notices and model cards are in [core/delivery/licenses/](core/delivery/licenses/). Model weights in the release are the preserved fitted artifacts, with the same tensor values as the submitted package. The VideoMAE loader translates legacy q/v-bias names and supplies zero key biases as required by the runtime implementation.

## Ultralytics integration

The source code that calls Ultralytics is available in `core/scripts/`, including `extract_kuno_transfer_logits.py` and `produce_r34_window_rois.py`. The dependency is pinned in `environment/requirements-docker.txt`; its corresponding upstream source is linked above. The AGPL text is included in [Ultralytics_LICENSE.txt](core/delivery/licenses/Ultralytics_LICENSE.txt).

Apache-2.0 applies to our original contributions. It does not relicense Ultralytics or remove the AGPL obligations of a combined derivative work. Anyone redistributing or serving a combined work must comply with the applicable AGPL source and notice requirements. This release provides the full KURONAME implementation and the dependency source locations; it does not claim an enterprise Ultralytics license.

## Data attribution

The CUHK-X dataset was created by the AIoT Lab, Department of Information Engineering, The Chinese University of Hong Kong. Cite the CUHK-X paper listed in the report when using the dataset. Original media, participant files, sensor streams and QA tables must be obtained from the organizers. Our reference CSVs are model predictions, not Test ground-truth labels.
