#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./build_lm.sh C/filename.c [C/other.c ...] [lm]
#
# Example:
#   ./build_lm.sh C/sm_all_int_lut.c
#   ./build_lm.sh C/sm_float32.c lm
#   ./build_lm.sh C/a.c C/b.c C/c.c lm

if [ $# -lt 1 ]; then
  echo "Usage: $0 <path_to_c_file> [<path_to_c_file> ...] [lm]"
  echo "Example:"
  echo "  $0 C/sm_all_int_lut.c"
  echo "  $0 C/sm_float32.c lm"
  echo "  $0 C/a.c C/b.c C/c.c lm"
  exit 1
fi

# Determine if last arg is the optional "lm" flag
LINK_MATH=""
if [ "${!#}" = "lm" ]; then
  LINK_MATH="lm"
  # remove last argument ("lm") from "$@"
  set -- "${@:1:$(($#-1))}"
fi

if [ $# -lt 1 ]; then
  echo "ERROR: No source files provided."
  exit 1
fi

OUT_DIR="./bin"
CC="riscv64-linux-gnu-gcc"

if ! command -v "$CC" >/dev/null 2>&1; then
  echo "ERROR: '$CC' not found in PATH."
  echo "Install it with: sudo apt install gcc-riscv64-linux-gnu"
  exit 1
fi

mkdir -p "$OUT_DIR"

# Collect source files and validate existence
SRC_FILES=("$@")
for f in "${SRC_FILES[@]}"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: Source file '$f' not found."
    exit 1
  fi
done

# If multiple source files are provided, link them into one output.
if [ "${#SRC_FILES[@]}" -gt 1 ]; then
  BASE_NAME=$(basename "${SRC_FILES[0]}" .c)
  OUT_FILE="$OUT_DIR/$BASE_NAME.riscv"
  echo "Linking ${SRC_FILES[*]} -> $OUT_FILE ..."
  if [ "$LINK_MATH" = "lm" ]; then
    echo "Linking with -lm ..."
    "$CC" "${SRC_FILES[@]}" -static -o "$OUT_FILE" -lm
  else
    "$CC" "${SRC_FILES[@]}" -static -o "$OUT_FILE"
  fi
  echo "Done. Output: $OUT_FILE"
  exit 0
fi

# Single source file: compile as before
SRC_FILE="${SRC_FILES[0]}"
BASE_NAME=$(basename "$SRC_FILE" .c)
OUT_FILE="$OUT_DIR/$BASE_NAME.riscv"

echo "Compiling $SRC_FILE -> $OUT_FILE ..."
if [ "$LINK_MATH" = "lm" ]; then
  echo "Linking with -lm ..."
  "$CC" "$SRC_FILE" -static -o "$OUT_FILE" -lm
else
  "$CC" "$SRC_FILE" -static -o "$OUT_FILE"
fi
echo "Done. Output: $OUT_FILE"
