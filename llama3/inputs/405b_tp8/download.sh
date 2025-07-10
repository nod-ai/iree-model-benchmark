#!/bin/bash

# Usage: ./download.sh

set -euo pipefail
set -x

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"
cd "${SCRIPT_DIR}"

readonly SEQ_LEN=2048
readonly STRIDE=32

readonly URL_PREFIX="https://sharkpublic.blob.core.windows.net/sharkpublic/halo-models/llm-dev/llama3_405b"
readonly PREFILL_PREFIX="${URL_PREFIX}/prefill_args_bs4_${SEQ_LEN}_stride_${STRIDE}_tp8"
readonly DECODE_PREFIX="${URL_PREFIX}/decode_args_bs4_${SEQ_LEN}_stride_${STRIDE}_tp8"

echo "Get bs4 ${SEQ_LEN} seq len fp8 args"

mkdir -p "args_bs4_${SEQ_LEN}"
cd "args_bs4_${SEQ_LEN}"

wget -q --show-progress "${PREFILL_PREFIX}/cs_f16_shard_0.npy" -O prefill_cs_fp16_shard_0.npy &
wget -q --show-progress "${PREFILL_PREFIX}/cs_f16_shard_1.npy" -O prefill_cs_fp16_shard_1.npy &
wget -q --show-progress "${PREFILL_PREFIX}/cs_f16_shard_2.npy" -O prefill_cs_fp16_shard_2.npy &
wget -q --show-progress "${PREFILL_PREFIX}/cs_f16_shard_3.npy" -O prefill_cs_fp16_shard_3.npy &
wget -q --show-progress "${PREFILL_PREFIX}/cs_f16_shard_4.npy" -O prefill_cs_fp16_shard_4.npy &
wget -q --show-progress "${PREFILL_PREFIX}/cs_f16_shard_5.npy" -O prefill_cs_fp16_shard_5.npy &
wget -q --show-progress "${PREFILL_PREFIX}/cs_f16_shard_6.npy" -O prefill_cs_fp16_shard_6.npy &
wget -q --show-progress "${PREFILL_PREFIX}/cs_f16_shard_7.npy" -O prefill_cs_fp16_shard_7.npy &
wget -q --show-progress "${PREFILL_PREFIX}/seq_block_ids.npy" -O prefill_seq_block_ids.npy &
wget -q --show-progress "${PREFILL_PREFIX}/seq_lens.npy" -O prefill_seq_lens.npy &
wget -q --show-progress "${PREFILL_PREFIX}/tokens.npy" -O prefill_tokens.npy # &
wget -q --show-progress "${DECODE_PREFIX}/cs_f16_shard_0.npy" -O decode_cs_fp16_shard_0.npy &
wget -q --show-progress "${DECODE_PREFIX}/cs_f16_shard_1.npy" -O decode_cs_fp16_shard_1.npy &
wget -q --show-progress "${DECODE_PREFIX}/cs_f16_shard_2.npy" -O decode_cs_fp16_shard_2.npy &
wget -q --show-progress "${DECODE_PREFIX}/cs_f16_shard_3.npy" -O decode_cs_fp16_shard_3.npy &
wget -q --show-progress "${DECODE_PREFIX}/cs_f16_shard_4.npy" -O decode_cs_fp16_shard_4.npy &
wget -q --show-progress "${DECODE_PREFIX}/cs_f16_shard_5.npy" -O decode_cs_fp16_shard_5.npy &
wget -q --show-progress "${DECODE_PREFIX}/cs_f16_shard_6.npy" -O decode_cs_fp16_shard_6.npy &
wget -q --show-progress "${DECODE_PREFIX}/cs_f16_shard_7.npy" -O decode_cs_fp16_shard_7.npy &
wget -q --show-progress "${DECODE_PREFIX}/seq_block_ids.npy" -O decode_seq_block_ids.npy &
wget -q --show-progress "${DECODE_PREFIX}/seq_lens.npy" -O decode_seq_lens.npy &
wget -q --show-progress "${DECODE_PREFIX}/next_tokens.npy" -O decode_next_tokens.npy &
wget -q --show-progress "${DECODE_PREFIX}/start_positions.npy" -O decode_start_positions.npy

echo Done
