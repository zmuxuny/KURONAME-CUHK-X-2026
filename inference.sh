#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: bash inference.sh <test_data_dir> <output_csv> [submission_id]" >&2; exit 2
fi
PACKAGE_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "${CUHKX_PYTHON:-python3}" "$PACKAGE_ROOT/src/predict.py" "$@"
