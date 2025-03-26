#!/bin/bash

# Usage: ./download.sh [128|8k|16k]

set -euo pipefail

if (( $# != 1 )); then
    echo "usage : $0 [128|8k|16k]"
    exit 1
fi

set -x

readonly SEQ_LEN="$1"

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"
cd "${SCRIPT_DIR}"

readonly URL_PREFIX="https://sharkpublic.blob.core.windows.net/sharkpublic/halo-models/llm-dev/llama3_8b"

echo "Get bs4 ${SEQ_LEN} seq len args"

mkdir -p "args_bs4_${SEQ_LEN}"
cd "args_bs4_${SEQ_LEN}"

URL="${URL_PREFIX}/prefill_decode_bs4_${SEQ_LEN}_args"

wget -q --show-progress "${URL}/prefill_token_ids.npy" &
wget -q --show-progress "${URL}/prefill_seq_lens.npy" &
wget -q --show-progress "${URL}/prefill_seq_block_ids.npy" &
wget -q --show-progress "${URL}/prefill_cache_state.npy" &

wget -q --show-progress "${URL}/decode_next_tokens.npy" &
wget -q --show-progress "${URL}/decode_seq_lens.npy" &
wget -q --show-progress "${URL}/decode_seq_block_ids.npy" &
wget -q --show-progress "${URL}/decode_start_positions.npy" &
wget -q --show-progress "${URL}/decode_cache_state.npy"
cd ..

echo Done
