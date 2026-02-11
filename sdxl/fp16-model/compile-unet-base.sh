#!/bin/bash

# Base unet compilation script. This is intended to be invoked by other scripts.
# Usage:
# ./compile-unet-base.sh <iree-compile-path> <gfxip> <input mlir> -o <output vmfb> [extra flags]

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

readonly DEFAULT_FLAGS=(
)

set -x

"$IREE_COMPILE" "$INPUT" \
    --iree-hal-target-backends=rocm \
    --iree-rocm-target="$CHIP" \
    --iree-rocm-bc-dir="$(hipconfig --rocmpath)/amdgcn/bitcode" \
    --iree-global-opt-propagate-transposes=true \
    --iree-opt-outer-dim-concat=true \
    --iree-opt-const-eval=false \
    --iree-opt-data-tiling=false \
    --iree-rocm-waves-per-eu=2 \
    --iree-vm-target-truncate-unsupported-floats \
    --iree-codegen-llvmgpu-use-vector-distribution \
    --iree-llvmgpu-enable-prefetch \
    --iree-dispatch-creation-enable-aggressive-fusion=true \
    --iree-dispatch-creation-enable-fuse-horizontal-contractions=true \
    --iree-opt-aggressively-propagate-transposes=true \
    --iree-execution-model=async-external \
    --iree-preprocessing-pass-pipeline='builtin.module(iree-preprocessing-transpose-convolution-pipeline, iree-preprocessing-pad-to-intrinsics)' \
    "$@"
