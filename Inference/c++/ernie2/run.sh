#!/bin/bash

set -xe

gpu_id="-1"
if [ $# -ge 1 ]; then
  gpu_id="$1"
fi

num_threads=20
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

# MODEL_DIR=/data/mgallus/Sander/bert_app/float_model/
# DATA_FILE=/data/mgallus/Sander/bert_app/100_ds

# MODEL_DIR=/home/mgallus/data/fp32_model
# MODEL_DIR=/home/mgallus/data/origin
# MODEL_DIR=/home/mgallus/src/Sander/Paddle/qat/ernie_int8_fc/
MODEL_DIR=/home/mgallus/src/Sander/Paddle/qat/transformed_qat_int8_model/
# MODEL_DIR=/home/mgallus/src/Sander/Paddle/qat/working_qat_ernie/transformed_qat_int8_model/
DATA_FILE=/home/mgallus/data/ernie_model_data/1.8w.bs1
# DATA_FILE=/home/mgallus/data/ernie_model_data/small_1.8
LABEL_FILE=/home/mgallus/data/ernie_model_data/label.xnli.dev

# small_1.8
# 1.8w.bs1

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

output_prediction=false
if [ $# -ge 6 ]; then
  output_prediction=$6
fi

if [ $# -ge 7 ]; then
  LABEL_FILE=$7
fi

 # prepend="env MKLDNN_VERBOSE=2"
# prepend="cgdb --args"

export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1

${prepend} ./build/inference --logtostderr \
    --model_dir=${MODEL_DIR} \
    --data=${DATA_FILE} \
    --label=${LABEL_FILE} \
    --repeat=${REPEAT} \
    --use_gpu=${USE_GPU} \
    --num_threads=${num_threads} \
    --output_prediction=${output_prediction} \
    --use_int8 \
    --iterations=0 \
    #  --remove_scale \
    # --debug \
    # --warmup_steps=1 \
    # --use_analysis=true \
