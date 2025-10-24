#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./build.sh C/filename.c
#
# Example:
#   ./build.sh C/sm_all_int_lut.c

if [ $# -ne 1 ]; then
  echo "Usage: $0 <path_to_c_file>"
  echo "Example: $0 C/sm_all_int_lut.c"
  exit 1
fi

SRC_FILE="$1"
OUT_DIR="./bin"
CC="riscv64-linux-gnu-gcc"

if ! command -v "$CC" >/dev/null 2>&1; then
  echo "ERROR: '$CC' not found in PATH."
  echo "Install it with: sudo apt install gcc-riscv64-linux-gnu"
  exit 1
fi

if [ ! -f "$SRC_FILE" ]; then
  echo "ERROR: Source file '$SRC_FILE' not found."
  exit 1
fi

mkdir -p "$OUT_DIR"

BASE_NAME=$(basename "$SRC_FILE" .c)
OUT_FILE="$OUT_DIR/$BASE_NAME.riscv"

echo "Compiling $SRC_FILE -> $OUT_FILE ..."
"$CC" "$SRC_FILE" -static -o "$OUT_FILE"

echo "Done. Output: $OUT_FILE"
