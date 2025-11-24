#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./benchmark-unet.sh <hip-device-id> <batch-size> [<irpa-path-prefix>]

set -euo pipefail
set -x

if (( $# != 2 && $# != 3 )); then
  echo "usage: $0 <hip-device-id> <batch-size> [<irpa-path-prefix>]"
  exit 1
fi

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
readonly WORKING_DIR="${WORKING_DIR:-${SCRIPT_DIR}/tmp}"
readonly IREE_BENCHMARK="$(which iree-benchmark-module)"
readonly IREE_TRACY_CAPTURE="$(which iree-tracy-capture)"
readonly HIP_DEVICE="$1"
readonly BATCH_SIZE="$2"
readonly USE_TRACY="${USE_TRACY:-0}"

if ! [[ "${BATCH_SIZE}" =~ ^(2|4|8|16|18)$ ]]; then
  echo "Allowed batch-sizes: 2, 4, 8, 16, 18"
  exit 1
fi

readonly INPUT_PATH="${SCRIPT_DIR}/unet_npys/unet_inputs_bs${BATCH_SIZE}"

INPUTS="--input=@${INPUT_PATH}/run_forward_input_0.npy \
--input=@${INPUT_PATH}/run_forward_input_1.npy \
--input=@${INPUT_PATH}/run_forward_input_2.npy \
--input=@${INPUT_PATH}/run_forward_input_3.npy \
--input=@${INPUT_PATH}/run_forward_input_4.npy \
--input=@${INPUT_PATH}/run_forward_input_5.npy \
--input=@${INPUT_PATH}/run_forward_input_6.npy \
--input=@${INPUT_PATH}/run_forward_input_7.npy"

# IRPA file: https://sharkpublic.blob.core.windows.net/sharkpublic/sdxl-scripts-weights/punet_fp8_weights.irpa
# Size: 2615300096
# md5sum: 42df7496dc012548e5fc1a198cb1161d
readonly IRPA_PATH_PREFIX="${3:-/data/amdshark}"
readonly IRPA="${IRPA_PATH_PREFIX}/punet_fp8_weights.irpa"

run_benchmark() {
  "$IREE_BENCHMARK" \
    --device="hip://${HIP_DEVICE}" \
    --device_allocator=caching \
    --module="${WORKING_DIR}/punet_bs${BATCH_SIZE}.vmfb" \
    --parameters=model="${IRPA}" \
    --function=run_forward \
    $INPUTS \
    --benchmark_repetitions=3
}

if (( "${USE_TRACY}" == "1" )); then
  TRACY_PORT=8087 IREE_PY_RUNTIME=tracy TRACY_NO_EXIT=1 run_benchmark &
  "${IREE_TRACY_CAPTURE}" -f -p 8087 -o "${WORKING_DIR}/punet_fp8_bs${BATCH_SIZE}.tracy"
else
  run_benchmark
fi
