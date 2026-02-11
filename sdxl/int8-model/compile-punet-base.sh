#!/bin/bash

# Base unet compilation script. This is intended to be invoked by other scripts.
# Usage:
# ./compile-unet-base.sh <iree-compile-path> <gfxip> <input mlir> [spec_file] -o <output vmfb> [extra flags]

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

SPEC_FILE=""
if [ $# -gt 3 ] && [[ "$4" != -* ]]; then
  SPEC_FILE="$(realpath "$4")"
  if [ ! -f "$SPEC_FILE" ] ; then
    echo "Specified spec file not found: ${SPEC_FILE}"
    exit 1
  fi
  shift 4
else
  shift 3
fi

readonly FLAGS=(
"--iree-preprocessing-pass-pipeline=builtin.module(util.func(iree-global-opt-raise-special-ops, iree-flow-canonicalize), iree-preprocessing-transpose-convolution-pipeline, iree-preprocessing-pad-to-intrinsics{pad-target-type=conv}, util.func(iree-preprocessing-generalize-linalg-matmul-experimental))"
)

set -x

CMD_ARGS=(
    "$IREE_COMPILE" "$INPUT"
    --iree-hal-target-backends=rocm
    --iree-rocm-target=$CHIP
    --iree-execution-model=async-external
    --iree-opt-level=O3
    --iree-vm-target-truncate-unsupported-floats
    --iree-codegen-llvmgpu-use-vector-distribution
    --iree-llvmgpu-enable-prefetch=1
    --iree-config-add-tuner-attributes
    "${FLAGS[@]}"
)

if [ -n "$SPEC_FILE" ]; then
    CMD_ARGS+=(--iree-codegen-transform-dialect-library="$SPEC_FILE")
fi

CMD_ARGS+=("$@")

"${CMD_ARGS[@]}"
