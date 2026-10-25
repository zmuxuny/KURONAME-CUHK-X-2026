# KURONAME frozen inference runtime

Primary Selected submission: 56239668 (Private 0.90294).
Secondary Selected submission: 56258762 (Private 0.90000).

Run `bash inference.sh /input /output/submission.csv` for the primary, or append `56258762` for the secondary. `/input` contains the official label-free Testing data and test_qa.csv; `/output` is writable. Use a fresh output filename for each run.

After the documented model download and Docker build, all deployed weights and inference dependencies are in this image. Run with GPU access and `--network none`. Test data and outputs are mounted at run time. Setup instructions, model terms, the technical report and reference predictions are available at https://github.com/zmuxuny/KURONAME-CUHK-X-2026.
