#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./benchmark-405b-fp4-decode.sh N [<irpa-path>]

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
readonly IREE_TRACY_CAPTURE="$(which iree-tracy-capture)"

readonly -a INPUTS=(
    "--input=4x1xi64"
    "--input=4xi64"
    "--input=4xi64"
    "--input=4x128xi64"
    "--input=128x8257536xf8E4M3FN"
)

readonly IRPA_PATH="${2:-/shark-dev/405b/instruct/weights/fp4/fp4_2025_07_10_fn.irpa}"

echo "Using IRPA file:"
stat -c "%y %s %n" "${IRPA_PATH}"

run_benchmark() {
    ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$IREE_BENCHMARK" \
	--device="hip://${HIP_DEVICE}" \
	--device_allocator=caching \
	--hip_use_streams=true \
	--module="${WORKING_DIR}/${PREFIX}.405b_fp4.vmfb" \
	--parameters=model="${IRPA_PATH}" \
	--function=decode_bs4 \
	"${INPUTS[@]}" \
	--benchmark_repetitions=3
}

if (( "${USE_TRACY}" == "1")); then
    TRACY_PORT=8087 IREE_PY_RUNTIME=tracy TRACY_NO_EXIT=1 run_benchmark &
    "${IREE_TRACY_CAPTURE}" -f -p 8087 -o "${WORKING_DIR}/${PREFIX}.decode.405b_fp4.tracy"
else
    run_benchmark
fi
