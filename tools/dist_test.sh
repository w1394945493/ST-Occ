#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
# PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

find_random_available_port() {
    while : ; do
        PORT=$(shuf -i 20000-65000 -n 1)
        ss -lpn | grep -q ":${PORT} " || break
    done
    echo ${PORT}
}
# Check if the PORT environment variable is already set
if [ -z "$PORT" ]; then
    # If PORT is not set, find a random available port
    PORT=$(find_random_available_port)
fi

echo "Using port: $PORT"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    ${@:4}
