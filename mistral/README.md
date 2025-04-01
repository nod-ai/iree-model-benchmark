# Mistral Nemo Scripts

## Instructions

1. Download IRPA.
   * fp16: `https://sharkpublic.blob.core.windows.net/sharkpublic/mistral_nemo.irpa` 
1. Compile:
   * fp16: `./compile-12-fp16.sh gfx942`
1. Benchmark prefill:
   * fp16: `./benchmark-12-prefill.sh 0 /data/irpa`
1. Benchmark decode:
   * fp16: `./benchmark-12-decode.sh 0 /data/irpa`
