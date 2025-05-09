#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./benchmark-8b-prefill-fp8.sh N [128|2048]

set -euo pipefail

if (( $# != 2 && $# != 3 )); then
  echo "usage: $0 <hip-device-id> <seq-length:[128|2048]> [<irpa-path>]"
  exit 1
fi

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"
readonly WORKING_DIR="${WORKING_DIR:-${SCRIPT_DIR}/tmp}"
readonly PREFIX="${PREFIX:-base}"
readonly IREE_BENCHMARK="$(which iree-benchmark-module)"
readonly HIP_DEVICE="$1"
readonly SEQ_LENGTH="$2"
readonly INPUT_PATH="${INPUT_PATH:-${SCRIPT_DIR}/inputs/8b_fp8/args_bs4_${SEQ_LENGTH}}"
readonly USE_TRACY="${USE_TRACY:-0}"
readonly IREE_TRACY_CAPTURE="$(which iree-tracy-capture)"

if (("${SEQ_LENGTH}" == "128")); then
    readonly TOKEN_ID_SHAPE="4x128xi64"
    readonly BLOCK_ID_SHAPE="4x4xi64"
elif (("${SEQ_LENGTH}" == "2048")); then
    readonly TOKEN_ID_SHAPE="4x2048xi64"
    readonly BLOCK_ID_SHAPE="4x64xi64"
fi

readonly -a INPUTS=(
    "--input=${TOKEN_ID_SHAPE}=@${INPUT_PATH}/prefill_token_ids_${TOKEN_ID_SHAPE}.bin"
    "--input=4xi64=@${INPUT_PATH}/prefill_seq_lens_4xi64.bin"
    "--input=${BLOCK_ID_SHAPE}=@${INPUT_PATH}/prefill_seq_block_ids_${BLOCK_ID_SHAPE}.bin"
    "--input=261x2097152xf8E4M3FNUZ=@${INPUT_PATH}/prefill_cache_state_261x2097152xf8E4M3FNUZ.bin"
)

# IRPA file:
# Size: 9081761792
# md5sum: da94d6e49c55322335374ac3e8585e32
readonly IRPA="${3:-/data/shark/8b_fp8.irpa}"

echo "Using IRPA file:"
stat -c "%y %s %n" "${IRPA}"
# md5sum "${IRPA}"

set -x

run_benchmark() {
  "$IREE_BENCHMARK" \
    --device="hip://${HIP_DEVICE}" \
    --device_allocator=caching \
    --hip_use_streams=true \
    --module="${WORKING_DIR}/${PREFIX}.8b_fp8.vmfb" \
    --parameters=model="${IRPA}" \
    --function=prefill_bs4 \
    "${INPUTS[@]}" \
    --benchmark_repetitions=3
}

if (( "${USE_TRACY}" == "1")); then
  IREE_PY_RUNTIME=tracy TRACY_NO_EXIT=1 run_benchmark &
  "${IREE_TRACY_CAPTURE}" -f -o "${WORKING_DIR}/${PREFIX}.8b_fp8.tracy"
else
  run_benchmark
fi
