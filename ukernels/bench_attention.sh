#!/bin/bash

for type in "bf16" "f16";
do
  for s0 in 128 137 4096 4099;
  do
    for gqa in 1;
    do
      [[ $gqa = 1 ]] && enable_gqa="" || enable_gqa="--gqa"
      cmd="python attention.py -c gfx942 -b 4 -m ${s0} -n 128 -k1 128 -k2 ${s0} --dynamic-dims m k2 --dtype ${type} ${enable_gqa}"
      echo
      echo
      echo "${cmd}"
      eval ${cmd}
    done
  done
done
