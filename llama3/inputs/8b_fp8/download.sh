#!/bin/bash

# Usage: ./download.sh [128]

set -euo pipefail

if (( $# != 1 )); then
    echo "usage : $0 [128|2048]"
    exit 1
fi

set -x

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"
cd "${SCRIPT_DIR}"

readonly SEQ_LEN="$1"

readonly URL_PREFIX="https://sharkpublic.blob.core.windows.net/sharkpublic/halo-models/llm-dev/llama3_8b/args_bs4_fp8_${SEQ_LEN}"

echo "Get bs4 ${SEQ_LEN} seq len fp8 args"

mkdir -p "args_bs4_${SEQ_LEN}"
cd "args_bs4_${SEQ_LEN}"

if (("${SEQ_LEN}" == "128")); then
    PREFILL_TOKEN_ID_SHAPE="4x128xi64"
    PREFILL_SEQ_BLOCK_ID_SHAPE="4x4xi64"
    DECODE_SEQ_BLOCK_ID_SHAPE="4x5xi64"
elif (("${SEQ_LEN}" == "2048")); then
    PREFILL_TOKEN_ID_SHAPE="4x2048xi64"
    PREFILL_SEQ_BLOCK_ID_SHAPE="4x64xi64"
    DECODE_SEQ_BLOCK_ID_SHAPE="4x65xi64"
else
    echo "unhandled seq length " ${SEQ_LEN}
    exit 1
fi

wget -q --show-progress "${URL_PREFIX}/prefill_token_ids_${PREFILL_TOKEN_ID_SHAPE}.bin" &
wget -q --show-progress "${URL_PREFIX}/prefill_seq_lens_4xi64.bin" &
wget -q --show-progress "${URL_PREFIX}/prefill_seq_block_ids_${PREFILL_SEQ_BLOCK_ID_SHAPE}.bin" &
wget -q --show-progress "${URL_PREFIX}/prefill_cache_state_261x2097152xf8E4M3FNUZ.bin" &
wget -q --show-progress "${URL_PREFIX}/decode_next_tokens_4x1xi64.bin" &
wget -q --show-progress "${URL_PREFIX}/decode_seq_block_ids_tensor_${DECODE_SEQ_BLOCK_ID_SHAPE}.bin" &
wget -q --show-progress "${URL_PREFIX}/decode_seq_lens_4xi64.bin" &
wget -q --show-progress "${URL_PREFIX}/decode_start_positions_4xi64.bin" &
wget -q --show-progress "${URL_PREFIX}/decode_cache_state_261x2097152xf8E4M3FNUZ.bin"

echo Done
