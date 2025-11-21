#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./runtest.sh bin/sm_float32.riscv

if [ $# -ne 1 ]; then
  echo "Usage: $0 <binary_path>"
  echo "Example: $0 bin/sm_float32.riscv"
  exit 1
fi

BIN_PATH="$1"
OUT_DIR="m5out/$(basename "$BIN_PATH" .riscv)"

mkdir -p "$OUT_DIR"

/work/build/RISCV/gem5.opt -d "$OUT_DIR" --debug-flags=Exec --debug-file=exec.log /work/src/softmax/run_config.py --binary "$BIN_PATH"
