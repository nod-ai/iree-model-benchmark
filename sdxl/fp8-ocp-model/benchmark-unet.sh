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
readonly BATCH_SIZE=1
readonly INPUT_PATH="${SCRIPT_DIR}/unet_npys/unet_inputs_bs${BATCH_SIZE}"

INPUTS="--input=@${INPUT_PATH}/run_forward_input_0.npy \
--input=@${INPUT_PATH}/run_forward_input_1.npy \
--input=@${INPUT_PATH}/run_forward_input_2.npy \
--input=@${INPUT_PATH}/run_forward_input_3.npy \
--input=@${INPUT_PATH}/run_forward_input_4.npy \
--input=@${INPUT_PATH}/run_forward_input_5.npy \
--input=@${INPUT_PATH}/run_forward_input_6.npy \
--input=@${INPUT_PATH}/run_forward_input_7.npy"

# IRPA file: https://sharkpublic.blob.core.windows.net/sharkpublic/sdxl-scripts-weights/stable_diffusion_xl_base_1_0_punet_dataset_fp8_ocp.irpa
# Size: 2615300096
# md5sum: 42df7496dc012548e5fc1a198cb1161d
readonly IRPA_PATH_PREFIX="${2:-/data/amdshark}"
readonly IRPA="${IRPA_PATH_PREFIX}/stable_diffusion_xl_base_1_0_punet_dataset_fp8_ocp.irpa"

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
  "${IREE_TRACY_CAPTURE}" -f -p 8087 -o "${WORKING_DIR}/punet_fp8.tracy"
else
  run_benchmark
fi
