#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./benchmark-unet.sh <hip-device-id> [<irpa-path-prefix>]

set -euo pipefail
set -x

if (( $# != 1 && $# != 2 )); then
  echo "usage: $0 <hip-device-id> [<irpa-path-prefix>]"
  exit 1
fi

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"
readonly WORKING_DIR="${WORKING_DIR:-${SCRIPT_DIR}/tmp}"
readonly IREE_BENCHMARK="$(which iree-benchmark-module)"
readonly IREE_TRACY_CAPTURE="$(which iree-tracy-capture)"
readonly HIP_DEVICE="$1"
readonly USE_TRACY="${USE_TRACY:-0}"

# IRPA file: https://sharkpublic.blob.core.windows.net/sharkpublic/sdxl-scripts-weights/sdxl_unet_int8_dataset.irpa
# Size: 2614669312
# md5sum: b9b2971e18d1dbcbbd0645263d8a8ac5
readonly IRPA_PATH_PREFIX="${2:-/data/shark}"
readonly IRPA="${IRPA_PATH_PREFIX}/sdxl_unet_int8_dataset.irpa"

run_benchmark() {
  "$IREE_BENCHMARK" \
    --device="hip://${HIP_DEVICE}" \
    --hip_use_streams=true \
    --hip_allow_inline_execution=true \
    --device_allocator=caching \
    --module="${WORKING_DIR}/punet.vmfb" \
    --parameters=model="${IRPA}" \
    --function=run_forward \
    --input=1x4x128x128xf16 \
    --input=2x64x2048xf16 \
    --input=2x1280xf16 \
    --input=2x6xf16 \
    --input=1xf16 \
    --input=1xsi64 \
    --input=100xf32 \
    --input=100xf32 \
    --benchmark_repetitions=3
}

if (( "${USE_TRACY}" == "1" )); then
  TRACY_PORT=8087 IREE_PY_RUNTIME=tracy TRACY_NO_EXIT=1 run_benchmark &
  "${IREE_TRACY_CAPTURE}" -f -p 8087 -o "${WORKING_DIR}/punet_int8.tracy"
else
  run_benchmark
fi
