#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./benchmark-8b-decode-fp8.sh 0 [128|2048]

set -euo pipefail

if (( $# != 2 && $# != 3 )); then
  echo "usage: $0 <hip-device-id> <seq-length:[128|2048]> [<ipra-path>]"
  exit 1
fi

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"
readonly WORKING_DIR="${WORKING_DIR:-${SCRIPT_DIR}/tmp}"
readonly PREFIX="${PREFIX:-base}"
readonly IREE_BENCHMARK="$(which iree-benchmark-module)"
readonly HIP_DEVICE="$1"
readonly SEQ_LENGTH="$2"
readonly INPUT_PATH="${INPUT_PATH:-${SCRIPT_DIR}/inputs/8b_fp8/args_bs4_128}"
readonly USE_TRACY="${USE_TRACY:-0}"
readonly IREE_TRACY_CAPTURE="$(which iree-tracy-capture)"

if (("${SEQ_LENGTH}" == "128")); then
    readonly BLOCK_ID_SHAPE="4x5xi64"
elif (("${SEQ_LENGTH}" == "2048")); then
    readonly BLOCK_ID_SHAPE="4x65xi64"
fi

readonly -a INPUTS=(
  "--input=4x1xi64=@${INPUT_PATH}/decode_next_tokens_4x1xi64.bin"
  "--input=4xi64=@${INPUT_PATH}/decode_seq_lens_4xi64.bin"
  "--input=4xi64=@${INPUT_PATH}/decode_start_positions_4xi64.bin"
  "--input=${BLOCK_ID_SHAPE}=@${INPUT_PATH}/decode_seq_block_ids_tensor_${BLOCK_ID_SHAPE}.bin"
  "--input=261x2097152xf8E4M3FNUZ=@${INPUT_PATH}/decode_cache_state_261x2097152xf8E4M3FNUZ.bin"
)

# IRPA file:
# Size: 16061181952
# md5sum: 8f4685d6799298609152dd509ba32e88
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
	--function=decode_bs4  \
	"${INPUTS[@]}" \
	--benchmark_repetitions=3
}

if (( "${USE_TRACY}" == "1")); then
    TRACY_PORT=8087 IREE_PY_RUNTIME=tracy TRACY_NO_EXIT=1 run_benchmark &
    "${IREE_TRACY_CAPTURE}" -f -p 8087 -o "${WORKING_DIR}/${PREFIX}.decode.8b_fp8.tracy"
else
    run_benchmark
fi
