#!/bin/bash

set -xe

gpu_id="-1"
if [ $# -ge 1 ]; then
  gpu_id="$1"
fi

num_threads=1
if [ $# -ge 2 ]; then
  num_threads=$2
fi

if [ ${gpu_id} -eq "-1" ]; then
  USE_GPU=false
  export CUDA_VISIBLE_DEVICES=""
else
  USE_GPU=true
  export CUDA_VISIBLE_DEVICES="${gpu_id}"
fi

MODEL_DIR=/home/mgallus/data/fp32_model
# MODEL_DIR=/home/mgallus/src/Sander/Paddle/qat/transformed_qat_int8_model
#DATA_FILE=/home/mgallus/data/ernie_model_data/100_ds
DATA_FILE=/home/mgallus/data/ernie_model_data/one_ds
REPEAT=1

if [ $# -ge 3 ]; then
  MODEL_DIR=$3
fi

if [ $# -ge 4 ]; then
  DATA_FILE=$4
fi

profile=1
if [ $# -ge 5 ]; then
  profile=$5
fi

print_outputs=true
if [ $# -ge 6 ]; then
  print_outputs=$6
fi

./build/inference --logtostderr \
    --model_dir=${MODEL_DIR} \
    --data=${DATA_FILE} \
    --repeat=${REPEAT} \
    --warmup_steps=1 \
    --use_gpu=${USE_GPU} \
    --num_threads=${num_threads} \
    --use_analysis=true \
    --print_outputs=${print_outputs} \
    --profile=${profile}
