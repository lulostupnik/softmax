#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./build.sh <path_to_c_file> [additional_c_files...]
#
# Example:
#   ./build.sh C/sm_all_int_lut.c
#   ./build.sh C/softmax_int8.c C/main.c

if [ $# -lt 1 ]; then
  echo "Usage: $0 <path_to_c_file> [additional_c_files...]"
  echo "Example: $0 C/softmax_int8.c C/main.c"
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

# Check all source files exist
for SRC in "$@"; do
  if [ ! -f "$SRC" ]; then
    echo "ERROR: Source file '$SRC' not found."
    exit 1
  fi
done

mkdir -p "$OUT_DIR"

BASE_NAME=$(basename "$1" .c)
OUT_FILE="$OUT_DIR/$BASE_NAME.riscv"
DUMP_FILE="$OUT_DIR/$BASE_NAME.dump"

echo "Compiling $@ -> $OUT_FILE ..."
"$CC" "$@" -static -O3 -march=rv64gc -o "$OUT_FILE"

echo "Generating objdump -> $DUMP_FILE ..."
riscv64-linux-gnu-objdump -d "$OUT_FILE" > "$DUMP_FILE"

echo "Done. Output: $OUT_FILE"
echo "      Objdump: $DUMP_FILE"
