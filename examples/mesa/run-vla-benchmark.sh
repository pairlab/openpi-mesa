#!/bin/bash
# Thin shim that runs vla-benchmark's eval_server_parallel.py inside its own
# pinned venv. The repo root and venv both live off-home on the project volume.
#
# Override VLA_BENCHMARK_ROOT / VLA_BENCHMARK_PYTHON if they live elsewhere.

set -euo pipefail

VLA_BENCHMARK_ROOT="${VLA_BENCHMARK_ROOT:-/storage/home/hcoda1/5/fchang40/vla-benchmark}"
VLA_BENCHMARK_PYTHON="${VLA_BENCHMARK_PYTHON:-/storage/project/r-agarg35-0/fchang40/venvs/vla-benchmark/bin/python}"

if [ ! -d "$VLA_BENCHMARK_ROOT" ]; then
    echo "[run-vla-benchmark.sh] vla-benchmark repo not found at $VLA_BENCHMARK_ROOT" >&2
    exit 2
fi
if [ ! -x "$VLA_BENCHMARK_PYTHON" ]; then
    echo "[run-vla-benchmark.sh] python not found at $VLA_BENCHMARK_PYTHON" >&2
    exit 2
fi

cd "$VLA_BENCHMARK_ROOT"
exec "$VLA_BENCHMARK_PYTHON" scripts/eval_server_parallel.py "$@"
