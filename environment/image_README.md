# KURONAME frozen inference runtime

Primary Selected submission: 56239668 (Private 0.90294).
Secondary Selected submission: 56258762 (Private 0.90000).

Run `bash inference.sh /input /output/submission.csv` for the primary, or append `56258762` for the secondary. `/input` contains the official label-free Testing data and test_qa.csv; `/output` is writable. Use a fresh output filename for each run.

All deployed weights and inference dependencies are in this image. Run with GPU access and `--network none`. Test data and outputs are mounted at run time. The accompanying verification ZIP provides the signed declaration, complete documentation, training provenance, selected Kaggle CSVs and measured validation records.
