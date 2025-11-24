#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./benchmark-unet.sh <hip-device-id> [<irpa-path-prefix>]

set -euo pipefail
set -x

if (( $# != 1 && $# != 2 )); then
  echo "usage: $0 <hip-device-id> [<irpa-path-prefix>]"
  exit 1
fi

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
readonly WORKING_DIR="${WORKING_DIR:-${SCRIPT_DIR}/tmp}"
readonly IREE_BENCHMARK="$(which iree-benchmark-module)"
readonly IREE_TRACY_CAPTURE="$(which iree-tracy-capture)"
readonly HIP_DEVICE="$1"
readonly USE_TRACY="${USE_TRACY:-0}"

# IRPA file: https://sharkpublic.blob.core.windows.net/sharkpublic/sdxl-scripts-weights/scheduled_unet_fp16.irpa
# Size: 5135167488
# md5sum: 4ca340fcea6533e0693bca895991f12c
readonly IRPA_PATH_PREFIX="${2:-/data/amdshark}"
readonly IRPA="${IRPA_PATH_PREFIX}/scheduled_unet_fp16.irpa"

run_benchmark() {
  "$IREE_BENCHMARK" \
    --device="hip://${HIP_DEVICE}" \
    --device_allocator=caching \
    --module="${WORKING_DIR}/unet.vmfb" \
    --parameters=model="${IRPA}" \
    --function=main \
    --input=1x4x128x128xf16 \
    --input=1xi64 \
    --input=2x64x2048xf16 \
    --input=2x1280xf16 \
    --input=2x6xf16 \
    --input=1xf16 \
    --benchmark_repetitions=3
}

if (( "${USE_TRACY}" == "1" )); then
  TRACY_PORT=8087 IREE_PY_RUNTIME=tracy TRACY_NO_EXIT=1 run_benchmark &
  "${IREE_TRACY_CAPTURE}" -f -p 8087 -o "${WORKING_DIR}/unet_fp16.tracy"
else
  run_benchmark
fi
