#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./benchmark-unet.sh N

set -euo pipefail

if (( $# != 1 && $# != 2 )); then
  echo "usage: $0 <hip-device-id> [<irpa-path>]"
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
  "--input=32x1xsi64"
  "--input=32xsi64"
  "--input=32xsi64"
  "--input=32x32xsi64"
  "--input=1024x2621440xf8E4M3FNUZ"
)

# IRPA file:
# Size: 16061181952
# md5sum: 8f4685d6799298609152dd509ba32e88
readonly IRPA="${2:-/data/Mistral-Nemo-Instruct-2407-FP8/quark_mistral_nemo.irpa}"

echo "Using IRPA file:"
stat -c "%y %s %n" "${IRPA}"
# md5sum "${IRPA}"

set -x

run_benchmark() {
    "$IREE_BENCHMARK" \
	--device="hip://${HIP_DEVICE}" \
	--device_allocator=caching \
	--hip_use_streams=true \
	--module="${WORKING_DIR}/${PREFIX}.vmfb" \
	--parameters=model="${IRPA}" \
	--function=decode_bs32 \
	"${INPUTS[@]}" \
	--benchmark_repetitions=3
}

if (( "${USE_TRACY}" == "1")); then
    TRACY_PORT=8087 IREE_PY_RUNTIME=tracy TRACY_NO_EXIT=1 run_benchmark &
    "${IREE_TRACY_CAPTURE}" -f -p 8087 -o "${WORKING_DIR}/${PREFIX}.tracy"
else
    run_benchmark
fi
