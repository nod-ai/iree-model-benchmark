#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./benchmark-405b-tp8-prefill.sh [<irpa-path>]

set -euo pipefail
set -x

if (( $# != 0 && $# != 1 )); then
  echo "usage: $0 [<irpa-path>]"
  exit 1
fi

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"
readonly WORKING_DIR="${WORKING_DIR:-${SCRIPT_DIR}/tmp}"
readonly PREFIX="${PREFIX:-base}"
readonly IREE_BENCHMARK="$(which iree-benchmark-module)"
readonly INPUT_PATH="${INPUT_PATH:-${SCRIPT_DIR/inputs/405b_tp8/args_bs4_2048}"
readonly USE_TRACY="${USE_TRACY:-0}"
readonly IREE_TRACY_CAPTURE="$(which iree-tracy-capture)"

readonly -a INPUTS=(
  "--input=@${INPUT_PATH}/prefill_tokens.npy"
  "--input=@${INPUT_PATH}/prefill_seq_lens.npy"
  "--input=@${INPUT_PATH}/prefill_seq_block_ids.npy"
  "--input=@${INPUT_PATH}/prefill_cs_f16_shard_0.npy"
  "--input=@${INPUT_PATH}/prefill_cs_f16_shard_1.npy"
  "--input=@${INPUT_PATH}/prefill_cs_f16_shard_2.npy"
  "--input=@${INPUT_PATH}/prefill_cs_f16_shard_3.npy"
  "--input=@${INPUT_PATH}/prefill_cs_f16_shard_4.npy"
  "--input=@${INPUT_PATH}/prefill_cs_f16_shard_5.npy"
  "--input=@${INPUT_PATH}/prefill_cs_f16_shard_6.npy"
  "--input=@${INPUT_PATH}/prefill_cs_f16_shard_7.npy"
)

readonly IRPA_PATH="${2:-/shark-dev/llama3.1/405b/instruct/weights/fp16/tp8}"

# Base IRPA file:
# Size : 524288
# md5sum: eb4357afbc1dd40dee1480b9e6120f36
BASE_IRPA="${IRPA_PATH}/llama3_405b_instruct_fp16.irpa"

echo "Using Base IRPA file:"
stat -c "%y %s %n" "${BASE_IRPA}"
# md5sum "${IRPA}"

readonly -a IRPAS=(
    "--parameters=model=${BASE_IRPA}"
    "--parameters=model=${IRPA_PATH}/llama3_405b_instruct_fp16.rank0.irpa"
    "--parameters=model=${IRPA_PATH}/llama3_405b_instruct_fp16.rank1.irpa"
    "--parameters=model=${IRPA_PATH}/llama3_405b_instruct_fp16.rank2.irpa"
    "--parameters=model=${IRPA_PATH}/llama3_405b_instruct_fp16.rank3.irpa"
    "--parameters=model=${IRPA_PATH}/llama3_405b_instruct_fp16.rank4.irpa"
    "--parameters=model=${IRPA_PATH}/llama3_405b_instruct_fp16.rank5.irpa"
    "--parameters=model=${IRPA_PATH}/llama3_405b_instruct_fp16.rank6.irpa"
    "--parameters=model=${IRPA_PATH}/llama3_405b_instruct_fp16.rank7.irpa"
    )

run_benchmark() {
    ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$IREE_BENCHMARK" \
	--device="hip://0" \
	--device="hip://1" \
	--device="hip://2" \
	--device="hip://3" \
	--device="hip://4" \
	--device="hip://5" \
	--device="hip://6" \
	--device="hip://7" \
	--device_allocator=caching \
	--hip_use_streams=true \
	--module="${WORKING_DIR}/${PREFIX}.405b_fp16_tp8.vmfb" \
	"${IRPAS[@]}" \
	--function=prefill_bs4 \
	"${INPUTS[@]}" \
	--benchmark_repetitions=3
}

if (( "${USE_TRACY}" == "1")); then
    TRACY_PORT=8087 IREE_PY_RUNTIME=tracy TRACY_NO_EXIT=1 run_benchmark &
    "${IREE_TRACY_CAPTURE}" -f -p 8087 -o "${WORKING_DIR}/${PREFIX}.prefill.405b_tp8.tracy"
else
    run_benchmark
fi
