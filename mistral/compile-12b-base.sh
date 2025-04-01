#!/bin/bash

# Base llama 12b compilation script. This is intended to be invoked by other scripts.
# Usage:
# ./compile-12b-base.sh <iree-compile-path> <gfxip> <attention_matmul_spec_file> <input mlir> -o <output vmfb> [extra flags]

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"

readonly IREE_COMPILE="$(realpath "$1")"
if [ ! -f "$IREE_COMPILE" ] ; then
  echo "Specified iree-compile binary not found: ${IREE_COMPILE}"
  exit 1
fi

readonly CHIP="$2"

readonly INPUT="$(realpath "$3")"
if [ ! -f "$INPUT" ] ; then
  echo "Input mlir file not found: ${INPUT}"
  exit 1
fi

shift 3

set -x

mkdir -p tmp

"$IREE_COMPILE" "$INPUT" \
  --iree-hal-target-backends=rocm \
  --iree-hip-target=$CHIP \
  --iree-hal-target-device=hip \
  --iree-opt-level=O3 \
  --iree-hal-indirect-command-buffers=true \
  --iree-stream-resource-memory-model=discrete \
  --iree-hal-memoization=true \
  "$@"
