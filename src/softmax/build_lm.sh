#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./build.sh C/filename.c [lm]
#
# Example:
#   ./build.sh C/sm_all_int_lut.c
#   ./build.sh C/sm_float32.c lm

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  echo "Usage: $0 <path_to_c_file> [lm]"
  echo "Example:"
  echo "  $0 C/sm_all_int_lut.c"
  echo "  $0 C/sm_float32.c lm"
  exit 1
fi

SRC_FILE="$1"
OUT_DIR="./bin"
CC="riscv64-linux-gnu-gcc"
LINK_MATH=${2:-""}   # optional second argument

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

# Build command
if [ "$LINK_MATH" = "lm" ]; then
  echo "Linking with -lm ..."
  "$CC" "$SRC_FILE" -static -o "$OUT_FILE" -lm
else
  "$CC" "$SRC_FILE" -static -o "$OUT_FILE"
fi

echo "Done. Output: $OUT_FILE"
