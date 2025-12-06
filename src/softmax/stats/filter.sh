#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_file>"
    exit 1
fi

INPUT="$1"

# Extract directory and filename
INPUT_DIR=$(dirname "$INPUT")
BASENAME=$(basename "$INPUT")

# Output folder next to input file
OUTPUT_DIR="${INPUT_DIR}/output"
mkdir -p "$OUTPUT_DIR"

OUTPUT_FILE="${OUTPUT_DIR}/${BASENAME}"

# Pattern without trailing backslashes (single-line | separated)
PATTERN='^(simTicks|board\.processor\.cores\.core\.statFuBusy::IntAlu|board\.processor\.cores\.core\.statFuBusy::DMUpdateOp|board\.processor\.cores\.core\.statFuBusy::IntMult|board\.processor\.cores\.core\.statFuBusy::IntDiv|board\.processor\.cores\.core\.statIssuedInstType_0::IntAlu|board\.processor\.cores\.core\.statIssuedInstType_0::DMUpdateOp|board\.processor\.cores\.core\.statIssuedInstType_0::IntMult|board\.processor\.cores\.core\.statIssuedInstType_0::IntDiv|board\.processor\.cores\.core\.commit\.committedInstType_0::IntAlu|board\.processor\.cores\.core\.commit\.committedInstType_0::DMUpdateOp|board\.processor\.cores\.core\.commit\.committedInstType_0::IntMult|board\.processor\.cores\.core\.commit\.committedInstType_0::IntDiv|board\.processor\.cores\.core\.commitStats0\.committedInstType::IntAlu|board\.processor\.cores\.core\.commitStats0\.committedInstType::DMUpdateOp|board\.processor\.cores\.core\.commitStats0\.committedInstType::IntMult|board\.processor\.cores\.core\.commitStats0\.committedInstType::IntDiv)'

# Apply filtering
grep -E "$PATTERN" "$INPUT" > "$OUTPUT_FILE"

echo "Filtered file written to: $OUTPUT_FILE"
