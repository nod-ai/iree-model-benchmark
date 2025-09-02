#!/bin/bash

# Usage: ./download.sh

set -euo pipefail
set -x

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"
cd "${SCRIPT_DIR}"

readonly TOKEN_LEN=2500

readonly URL_PREFIX="https://sharkpublic.blob.core.windows.net/sharkpublic/halo-models/llm-dev/llama3_405b/fp4/inputs/real_inputs"
readonly PREFILL_PREFIX="${URL_PREFIX}/prefill/${TOKEN_LEN}"
readonly DECODE_PREFIX="${URL_PREFIX}/decode/${TOKEN_LEN}"

echo "Get bs4 ${TOKEN_LEN} token len fp4 args"

mkdir -p "args_bs4_${TOKEN_LEN}"
cd "args_bs4_${TOKEN_LEN}"

wget -q --show-progress "${PREFILL_PREFIX}/prefill_bs4_tokenlen${TOKEN_LEN}_input0_tokens.npy" -O prefill_input0_tokens.npy
wget -q --show-progress "${PREFILL_PREFIX}/prefill_bs4_tokenlen${TOKEN_LEN}_input1_seq_lens.npy" -O prefill_input1_seq_lens.npy
wget -q --show-progress "${PREFILL_PREFIX}/prefill_bs4_tokenlen${TOKEN_LEN}_input2_seq_block_ids.npy" -O prefill_input2_seq_block_ids.npy
wget -q --show-progress "${PREFILL_PREFIX}/prefill_bs4_tokenlen${TOKEN_LEN}_input3_kv_cache_state.npy" -O prefill_input3_kv_cache_state.npy

wget -q --show-progress "${DECODE_PREFIX}/decode_bs4_tokenlen${TOKEN_LEN}_input0_tokens.npy" -O decode_input0_tokens.npy
wget -q --show-progress "${DECODE_PREFIX}/decode_bs4_tokenlen${TOKEN_LEN}_input1_seq_lens.npy" -O decode_input1_seq_lens.npy
wget -q --show-progress "${DECODE_PREFIX}/decode_bs4_tokenlen${TOKEN_LEN}_input2_start_positions.npy" -O decode_input2_start_positions.npy
wget -q --show-progress "${DECODE_PREFIX}/decode_bs4_tokenlen${TOKEN_LEN}_input3_seq_block_ids.npy" -O decode_input3_seq_block_ids.npy
wget -q --show-progress "${DECODE_PREFIX}/decode_bs4_tokenlen${TOKEN_LEN}_input4_kv_cache_state.npy" -O decode_input4_kv_cache_state.npy

echo Done
