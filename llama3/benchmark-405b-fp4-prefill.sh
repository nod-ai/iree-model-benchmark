#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./benchmark-405b-fp4-prefill.sh N [<irpa-path>]

set -euo pipefail
set -x

if (( $# != 1 && $# != 2 )); then
  echo "usage: $0 <device number> [<irpa-path>]"
  exit 1
fi

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"
readonly WORKING_DIR="${WORKING_DIR:-${SCRIPT_DIR}/tmp}"
readonly PREFIX="${PREFIX:-base}"
readonly IREE_BENCHMARK="$(which iree-benchmark-module)"
readonly HIP_DEVICE="$1"
readonly USE_TRACY="${USE_TRACY:-0}"
readonly TRACY_CAPTURE="${TRACY_CAPTURE:-$(which iree-tracy-capture)}"
readonly TRACY_PORT=${TRACY_PORT:-8087}
readonly TOKEN_LEN="${TOKEN_LEN:-2500}"

readonly INPUT_PATH="${INPUT_PATH:-${SCRIPT_DIR}/inputs/405b_fp4/args_bs4_${TOKEN_LEN}}"

readonly -a INPUTS=(
  "--input=@${INPUT_PATH}/prefill_input0_tokens.npy"
  "--input=@${INPUT_PATH}/prefill_input1_seq_lens.npy"
  "--input=@${INPUT_PATH}/prefill_input2_seq_block_ids.npy"
  "--input=@${INPUT_PATH}/prefill_input3_kv_cache_state.npy"
)

readonly IRPA_PATH="${2:-/shark-dev/llama3.1/405b/instruct/weights/fp4/fp4_2025_07_10_fn.irpa}"

echo "Using IRPA file:"
stat -c "%y %s %n" "${IRPA_PATH}"

run_benchmark() {
    ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$IREE_BENCHMARK" \
	--device="hip://${HIP_DEVICE}" \
	--device_allocator=caching \
	--hip_use_streams=true \
	--module="${WORKING_DIR}/${PREFIX}.405b_fp4.vmfb" \
	--parameters=model="${IRPA_PATH}" \
	--function=prefill_bs4 \
	"${INPUTS[@]}" \
	--benchmark_repetitions=3
}

if (( "${USE_TRACY}" == "1")); then
    IREE_PY_RUNTIME=tracy TRACY_NO_EXIT=1 run_benchmark &
    "${TRACY_CAPTURE}" -f -p ${TRACY_PORT} -o "${WORKING_DIR}/${PREFIX}.prefill.405b_fp4.tracy"
else
    run_benchmark
fi
