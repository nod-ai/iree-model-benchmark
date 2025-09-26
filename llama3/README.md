# Llama3 Scripts

## Instructions

1. Download IRPA.
1. Fetch inputs:
   * fp16: `inputs/8b_fp16/download.sh 128`
   * fp8: `inputs/8b_fp8/download.sh [128|2048]`
   * 405b_fp4: `inputs/405b_fp4/download.sh`
1. Compile:
   * fp16: `./compile-8b-fp16.sh gfx942`
   * fp8: `./compile-8b-fp8.sh gfx942`
   * 405b_fp4 with asm kernel: `./compile-405b-fp4.sh gfx950`
   * 405b_fp4 with mlir kernel: `MLIR_FILE=405b_fp4 ./compile-405b-fp4.sh gfx950`
1. Benchmark prefill:
   * fp16: `./benchmark-8b-prefill.sh 0 /data/irpa`
   * fp8: `./benchmark-8b-prefill-fp8.sh 0 [128|2048] /data/irpa`
   * 405b_fp4: `./benchmark-405b-fp4-prefill.sh 0 /data/irpa`
1. Benchmark decode:
   * fp16: `./benchmark-8b-decode.sh 0 /data/irpa`
   * fp8: `./benchmark-8b-decode-fp8.sh 0 [128|2048] /data/irpa`
   * 405b_fp4: `./benchmark-405b-fp4-decode.sh 0 /data/irpa`
