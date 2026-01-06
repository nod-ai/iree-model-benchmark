#!/bin/bash

# Usage: ./download.sh
#
# The KV cache inputs must be saved with dtype 'V1' (void) for IREE compatibility.
# If inputs are uint8, convert with:
#   python3 -c "import numpy as np; arr=np.load('file.npy'); np.save('file.npy', arr.view(np.dtype('V1')))"

set -euo pipefail
set -x

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"
cd "${SCRIPT_DIR}"

readonly URL_PREFIX="https://sharkpublic.blob.core.windows.net/sharkpublic/halo-models/llm-dev/llama3_70b/fp4/inputs/real_inputs"

echo "Get 70B FP4 inputs from blob storage"

mkdir -p "args_bs4_2048"
cd "args_bs4_2048"

wget -q --show-progress "${URL_PREFIX}/prefill_input0_tokens.npy" -O prefill_input0_tokens.npy
wget -q --show-progress "${URL_PREFIX}/prefill_input1_seq_lens.npy" -O prefill_input1_seq_lens.npy
wget -q --show-progress "${URL_PREFIX}/prefill_input2_seq_block_ids.npy" -O prefill_input2_seq_block_ids.npy
wget -q --show-progress "${URL_PREFIX}/prefill_input3_kv_cache_state.npy" -O prefill_input3_kv_cache_state.npy

wget -q --show-progress "${URL_PREFIX}/decode_input0_tokens.npy" -O decode_input0_tokens.npy
wget -q --show-progress "${URL_PREFIX}/decode_input1_seq_lens.npy" -O decode_input1_seq_lens.npy
wget -q --show-progress "${URL_PREFIX}/decode_input2_start_positions.npy" -O decode_input2_start_positions.npy
wget -q --show-progress "${URL_PREFIX}/decode_input3_seq_block_ids.npy" -O decode_input3_seq_block_ids.npy
wget -q --show-progress "${URL_PREFIX}/decode_input4_kv_cache_state.npy" -O decode_input4_kv_cache_state.npy

echo Done
