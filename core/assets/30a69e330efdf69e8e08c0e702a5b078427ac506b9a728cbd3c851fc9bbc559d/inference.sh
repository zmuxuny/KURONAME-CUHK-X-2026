#!/usr/bin/env bash
set -euo pipefail
CUHKX_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "--experts" ]]; then
    shift
    exec "${CUHKX_PYTHON:-python}" "${CUHKX_ROOT}/scripts/infer_experts.py" "$@"
fi
exec "${CUHKX_PYTHON:-python}" "${CUHKX_ROOT}/scripts/inference.py" "$@"
