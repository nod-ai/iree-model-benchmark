#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./compile-unet.sh <target-chip> [spec_file] [extra flags]

set -euo pipefail

if (( $# < 1 )); then
  echo "usage: $0 <target-chip> [spec_file] [extra flags]"
  exit 1
fi

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"

readonly IREE_COMPILE="$(which iree-compile)"
readonly CHIP="$1"
WORKING_DIR="${WORKING_DIR:-${SCRIPT_DIR}}"

SPEC_FILE=""
if [ $# -gt 1 ] && [[ "$2" != -* ]]; then
  SPEC_FILE="$2"
  shift 2
else
  shift
fi

set -x

CMD_ARGS=(
  "${SCRIPT_DIR}/compile-punet-base.sh" "$IREE_COMPILE" "$CHIP"
  "${SCRIPT_DIR}/base_ir/stable_diffusion_xl_base_1_0_scheduled_unet_bs1_64_1024x1024_i8.mlir"
)

if [ -n "$SPEC_FILE" ]; then
  CMD_ARGS+=("$SPEC_FILE")
fi

CMD_ARGS+=(
  --iree-hal-dump-executable-configurations-to="${WORKING_DIR}/tmp/configurations"
  --iree-hal-dump-executable-intermediates-to="${WORKING_DIR}/tmp/intermediates"
  --iree-hal-dump-executable-sources-to="${WORKING_DIR}/tmp/sources"
  --iree-hal-dump-executable-binaries-to="${WORKING_DIR}/tmp/binaries"
  --iree-hal-dump-executable-benchmarks-to="${WORKING_DIR}/tmp/benchmarks"
  --iree-scheduling-dump-statistics-file="${WORKING_DIR}/tmp/punet_scheduling_stats.txt"
  --iree-scheduling-dump-statistics-format=csv
  -o "${WORKING_DIR}/tmp/punet.vmfb"
  "$@"
)

"${CMD_ARGS[@]}"
