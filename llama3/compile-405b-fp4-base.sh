#!/bin/bash

# Base llama 405b-fp4 compilation script. This is intended to be invoked by other scripts.
# Usage:
# ./compile-405b-fp4-base.sh <iree-compile-path> <gfxip> <input mlir> -o <output vmfb> [extra flags]

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"

readonly IREE_COMPILE="$(realpath "$1")"
if [ ! -f "$IREE_COMPILE" ] ; then
  echo "Specified iree-compile binary not found: ${IREE_COMPILE}"
  exit 1
fi
readonly USE_TRACY="${USE_TRACY:-0}"

readonly CHIP="$2"

readonly INPUT="$(realpath "$3")"
if [ ! -f "$INPUT" ] ; then
  echo "Input mlir file not found: ${INPUT}"
  exit 1
fi

shift 3

set -x

COMPILER_FLAGS=(
    "--iree-hip-target=$CHIP" \
    "--iree-hal-target-backends=rocm" \
    "--iree-opt-level=O3" \
    "--iree-dispatch-creation-propagate-collapse-across-expands=true" \
    "--iree-codegen-enable-default-tuning-specs=true" \
    "--iree-hal-indirect-command-buffers=true" \
    "--iree-stream-resource-memory-model=discrete" \
    "--iree-hip-specialize-dispatches" \
    "--iree-hal-memoization=true" \
    "--iree-stream-affinity-solver-max-iterations=1024")

if (( "${USE_TRACY}" == "1")); then
    COMPILER_FLAGS+=("--iree-hal-executable-debug-level=3")
fi

"$IREE_COMPILE" "$INPUT" "${COMPILER_FLAGS[@]}" "$@"
